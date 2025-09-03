from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Mapping, Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin

from .kernels import BaseKernel, KernelType, Linear
from .quadprog import quadprog


@dataclass
class SVM(BaseEstimator, ClassifierMixin):
    """Support Vector Machine solved via the dual QP.

    Parameters
    ----------
    C:
        Regularization parameter. Larger values try to fit the training data
        more exactly at the cost of a smaller margin.
    kernel:
        Specification of the kernel to use.  This can be an instance of a
        :class:`~sawmil.kernels.BaseKernel`, a callable, or a string understood
        by :func:`~sawmil.kernels.get_kernel` (e.g. ``"linear"`` or
        ``"rbf"``).
    solver:
        Name of the quadratic program solver backend.  ``"gurobi"`` and
        ``"osqp"`` are supported.
    tol:
        Threshold used to decide whether a Lagrange multiplier is treated as
        zero when identifying support vectors.
    verbose:
        If ``True`` the underlying solver may print progress information.
    """

    C: float = 1.0
    kernel: KernelType = field(default_factory=Linear)
    solver: str = "gurobi"
    tol: float = 1e-6
    verbose: bool = False
    solver_params: Optional[Mapping[str, Any]] = None


    # Fitted attributes -------------------------------------------------
    _k: Optional[BaseKernel] = field(default=None, init=False, repr=False)
    X_: Optional[npt.NDArray[np.float64]] = field(default=None, init=False, repr=False)
    y_: Optional[npt.NDArray[np.float64]] = field(default=None, init=False, repr=False)
    classes_: Optional[npt.NDArray[np.float64]] = field(default=None, init=False, repr=False)
    alpha_: Optional[npt.NDArray[np.float64]] = field(default=None, init=False, repr=False)
    support_: Optional[npt.NDArray[np.int64]] = field(default=None, init=False, repr=False)
    support_vectors_: Optional[npt.NDArray[np.float64]] = field(default=None, init=False, repr=False)
    dual_coef_: Optional[npt.NDArray[np.float64]] = field(default=None, init=False, repr=False)
    intercept_: Optional[float] = field(default=None, init=False, repr=False)
    coef_: Optional[npt.NDArray[np.float64]] = field(default=None, init=False, repr=False)


    def _get_kernel(self, X_train: npt.NDArray[np.float64]) -> BaseKernel:
        """Instantiate and fit kernel on training X for defaults like gamma."""
        if self._k is None:
            self._k = self.kernel
            self._k.fit(X_train)
        return self._k

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> "SVM":
        '''Fit the model to the training data.'''
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

        n = X.shape[0]
        f = -np.ones(n, dtype=float)
        Aeq = Y.reshape(1, -1)
        beq = np.array([0.0], dtype=float)
        lb = np.zeros(n, dtype=float)
        ub = np.full(n, float(self.C), dtype=float)

        # Solve dual
        alpha, _ = quadprog(H, f, Aeq, beq, lb, ub, verbose=self.verbose, solver=self.solver, solver_params=self.solver_params)
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
        '''Compute the decision function for the given bags.'''
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
        '''Predict the labels for the given bags.'''
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(float)

    def score(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        '''Compute the accuracy of the model on the given bags.'''
        y = np.asarray(y).ravel()
        yhat = self.predict(X)
        return float(np.mean(yhat == y))
