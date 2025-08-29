from __future__ import annotations
from typing import Optional
from sklearn.base import BaseEstimator
import numpy as np
import numpy.typing as npt

from .kernels import get_kernel, BaseKernel, KernelType, Linear
from .quadprog import quadprog_gurobi

class SVM(BaseEstimator):
    """Support Vector Machine (dual QP w/ Gurobi). Binary only."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: KernelType = "linear",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-6,       # threshold to detect support vectors
        verbose: bool = False,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.verbose = verbose

        # Fitted attributes
        self._k: Optional[BaseKernel] = None # the fitted kernel instance
        self.X_: Optional[npt.NDArray[np.float64]] = None
        self.y_: Optional[npt.NDArray[np.float64]] = None   # mapped to {-1,+1}
        self.classes_: Optional[npt.NDArray[np.float64]] = None  # original labels
        self.alpha_: Optional[npt.NDArray[np.float64]] = None
        self.support_: Optional[npt.NDArray[np.int64]] = None
        self.support_vectors_: Optional[npt.NDArray[np.float64]] = None
        self.dual_coef_: Optional[npt.NDArray[np.float64]] = None  # shape (1, n_SV)
        self.intercept_: Optional[float] = None
        self.coef_: Optional[npt.NDArray[np.float64]] = None  # only for linear kernel

    def _get_kernel(self, X_train: npt.NDArray[np.float64]) -> BaseKernel:
        """Instantiate and fit kernel on training X for defaults like gamma."""
        if self._k is None:
            self._k = get_kernel(self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
            self._k.fit(X_train)
        return self._k

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> "SVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D with len(y) == n_samples.")

        # Map arbitrary binary labels to {-1,+1} (stable, order by np.unique)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("Binary classification only (exactly two classes required).")
        self.classes_ = classes.astype(float)
        y_mapped = np.where(y == classes[0], -1.0, 1.0)

        self.X_ = X
        self.y_ = y_mapped

        # Build dual QP pieces
        k = self._get_kernel(X)
        K = k(X, X)  # (n,n)
        Y = y_mapped
        H = (Y[:, None] * Y[None, :]) * K
        H = 0.5 * (H + H.T)  # symmetrize

        n = X.shape[0]
        f = -np.ones(n, dtype=float)
        Aeq = Y.reshape(1, -1)
        beq = np.array([0.0], dtype=float)
        lb = np.zeros(n, dtype=float)
        ub = np.full(n, float(self.C), dtype=float)

        # Solve dual
        alpha, _ = quadprog_gurobi(H, f, Aeq, beq, lb, ub, verbose=self.verbose)
        self.alpha_ = alpha

        # Support vectors
        sv_mask = alpha > self.tol
        self.support_ = np.flatnonzero(sv_mask).astype(int)
        self.support_vectors_ = X[sv_mask]
        self.dual_coef_ = (alpha[sv_mask] * Y[sv_mask]).reshape(1, -1)

        # Intercept b using margin SVs: 0 < α_i < C
        on_margin = (alpha > self.tol) & (alpha < self.C - self.tol)
        if not np.any(on_margin):  # degenerate case: use all SVs
            on_margin = sv_mask
        b_vals = Y[on_margin] - (alpha * Y) @ K[:, on_margin]
        self.intercept_ = float(np.mean(b_vals)) if b_vals.size else 0.0

        # Linear primal weights if kernel is strictly linear
        self.coef_ = None
        if isinstance(k, Linear):
            self.coef_ = (alpha * Y) @ X  # shape (n_features,)

        return self

    def decision_function(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.X_ is None or self.alpha_ is None or self.y_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        # Fast path for linear kernel
        if self.coef_ is not None:
            return (X @ self.coef_) + self.intercept_
        k = self._get_kernel(self.X_)   # do NOT refit on X; use the training-fitted kernel
        Ktest = k(self.X_, X)           # (n_train, n_test)
        return (self.alpha_ * self.y_) @ Ktest + self.intercept_

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        scores = self.decision_function(X)
        pred_sign = np.where(scores >= 0, 1.0, -1.0)
        if self.classes_ is None:
            return pred_sign
        y0, y1 = self.classes_
        return np.where(pred_sign < 0, y0, y1)

    def score(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        y = np.asarray(y).ravel()
        yhat = self.predict(X)
        return float(np.mean(yhat == y))
