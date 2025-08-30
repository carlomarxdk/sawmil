# src/sawmil/sawmil.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin

from .bag import Bag, BagDataset
from .smil import sMIL
from .svm import SVM
from .kernels import KernelType, Linear

@dataclass
class sAwMIL(BaseEstimator, ClassifierMixin):
    C: float = 1.0
    kernel: KernelType = "Linear"
    # bag-kernel options used inside sMIL (stage 1)
    normalizer: str = "none"   # recommend "none" for sMIL
    p: float = 1.0
    fast_linear: bool = True

    # sMIL-specific scaling of C by block sizes
    smil_scale_C: bool = True  # <-- renamed; avoids duplicate field name

    tol: float = 1e-8
    verbose: bool = False

    # solver for the instance SVM in stage 2
    solver: str = "gurobi"

    # selection hyperparams
    eta: float = 0.1
    min_pos_ratio: float = 0.05

    # learned
    smil_: sMIL | None = None
    sil_: SVM | None = None
    classes_: npt.NDArray[np.float64] | None = None
    coef_: npt.NDArray[np.float64] | None = None
    intercept_: float | None = None
    cutoff_: float | None = None

    # ---------- helpers ----------
    @staticmethod
    def _coerce_bags(
        bags: Sequence[Bag] | BagDataset | Sequence[np.ndarray],
        y: Optional[npt.NDArray[np.float64]] = None,
        intra_bag_labels: Optional[Sequence[np.ndarray]] = None,
    ) -> List[Bag]:
        if isinstance(bags, BagDataset):
            return list(bags.bags)
        if len(bags) > 0 and isinstance(bags[0], Bag):  # type: ignore[index]
            return list(bags)  # type: ignore[return-value]
        # Raw arrays path: require y and intra_bag_labels
        if y is None or intra_bag_labels is None:
            raise ValueError("For raw arrays, pass both y and intra_bag_labels (one 1D mask per bag).")
        if not (len(bags) == len(y) == len(intra_bag_labels)):  # type: ignore[arg-type]
            raise ValueError("bags, y, and intra_bag_labels must have the same length.")
        out: List[Bag] = []
        for X_i, y_i, m_i in zip(bags, y, intra_bag_labels):  # type: ignore[assignment]
            out.append(
                Bag(
                    X=np.asarray(X_i, dtype=float),
                    y=float(y_i),
                    intra_bag_label=np.asarray(m_i, dtype=float).ravel(),  # <-- singular name
                )
            )
        return out

    @staticmethod
    def _flatten(blist: List[Bag]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Xs, ys, ms, bi = [], [], [], []
        for i, b in enumerate(blist):
            if b.n == 0:
                continue
            Xs.append(b.X)
            ys.append(np.full(b.n, float(b.y), dtype=float))
            ms.append(b.mask.astype(float))
            bi.append(np.full(b.n, i, dtype=int))
        if not Xs:
            return (np.zeros((0, 0)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,), dtype=int))
        return (np.vstack(Xs), np.concatenate(ys), np.concatenate(ms), np.concatenate(bi))

    @staticmethod
    def _singletonize(X: np.ndarray, y: float) -> List[Bag]:
        return [Bag(X=X[j:j+1, :], y=y) for j in range(X.shape[0])]

    # ---------- core ----------
    def fit(
        self,
        bags: Sequence[Bag] | BagDataset | Sequence[np.ndarray],
        y: Optional[npt.NDArray[np.float64]] = None,
        intra_bag_labels: Optional[Sequence[np.ndarray]] = None,
    ) -> "sAwMIL":
        # 1) coerce input
        blist = self._coerce_bags(bags, y, intra_bag_labels)
        if not blist:
            raise ValueError("No bags provided.")

        # 2) sMIL (stage 1) — use its decision on singletons to rank instances
        smil = sMIL(
            C=self.C,
            kernel=self.kernel,
            normalizer=self.normalizer,
            p=self.p,
            use_intra_labels=True,            # you said you want to enforce using intra labels
            fast_linear=self.fast_linear,
            scale_C=self.smil_scale_C,        # <-- use the sMIL-specific flag
            tol=self.tol,
            verbose=self.verbose,
        )
        smil.fit(blist)
        self.smil_ = smil

        # 3) gather all instances
        X_all, y_bag, mask, _ = self._flatten(blist)
        if X_all.shape[0] == 0:
            # degenerate
            self.sil_ = SVM(C=self.C, kernel="linear", solver=self.solver, tol=self.tol, verbose=self.verbose)
            self.sil_.coef_, self.sil_.intercept_ = np.zeros((blist[0].d,)), 0.0
            self.coef_, self.intercept_, self.cutoff_ = self.sil_.coef_, self.sil_.intercept_, 0.0
            return self

        # split by bag label
        pos_inst = (y_bag > 0)
        X_pos = X_all[pos_inst]
        mask_pos = mask[pos_inst]
        X_neg = X_all[~pos_inst]

        # no positives? fall back to all negative labels for SIL
        if X_pos.shape[0] == 0:
            X_sil = X_neg
            y_sil = -np.ones(X_neg.shape[0], dtype=float)
            sil = SVM(C=self.C, kernel=self.kernel, solver=self.solver,
                      tol=self.tol, verbose=self.verbose)
            sil.fit(X_sil, y_sil)
            self.sil_ = sil
            self.coef_ = sil.coef_.ravel() if sil.coef_ is not None else None
            self.intercept_ = float(sil.intercept_) if sil.intercept_ is not None else None
            self.cutoff_ = float("-inf")
            return self

        # 4) score positive-bag instances with sMIL (as singleton bags)
        pos_singletons = self._singletonize(X_pos, y=+1.0)
        S_pos = smil.decision_function(pos_singletons).ravel()

        # 5) select top-eta under the intra-label mask
        eta = float(self.eta)
        eta = min(max(eta, 1e-9), 1.0)
        if S_pos.size == 0:
            q = float("-inf")
            chosen = np.zeros(0, dtype=bool)
        else:
            q = float(np.quantile(S_pos, 1.0 - eta, method="linear"))
            chosen = (S_pos >= q) & (mask_pos >= 0.5)
            # fallback: ensure at least min_pos_ratio positives overall
            min_needed = max(1, int(self.min_pos_ratio * len(S_pos)))
            if chosen.sum() < min_needed:
                k = min(len(S_pos), max(min_needed, int(round(eta * len(S_pos)))))
                topk = np.argsort(-S_pos)[:k]
                chosen = np.zeros_like(chosen)
                chosen[topk] = True

        self.cutoff_ = q

        # 6) build SIL dataset
        y_pos = np.full(X_pos.shape[0], -1.0, dtype=float)
        y_pos[chosen] = +1.0
        X_sil = np.vstack([X_neg, X_pos])
        y_sil = np.hstack([-np.ones(X_neg.shape[0], dtype=float), y_pos])

        # 7) train instance SVM (stage 2) — pass solver here
        sil = SVM(
            C=self.C,
            kernel=self.kernel,
            solver=self.solver,          
            tol=self.tol,
            verbose=self.verbose,
        )
        sil.fit(X_sil, y_sil)
        self.sil_ = sil

        self.coef_ = sil.coef_.ravel() if sil.coef_ is not None else None
        self.intercept_ = float(sil.intercept_) if sil.intercept_ is not None else None
        return self

    # ---------- inference ----------
    def decision_function(self, bags: Sequence[Bag] | BagDataset | Sequence[np.ndarray]) -> npt.NDArray[np.float64]:
        blist = self._coerce_bags(bags)
        if self.sil_ is None:
            raise RuntimeError("sAwMIL is not fitted.")
        scores = np.empty(len(blist), dtype=float)
        for i, b in enumerate(blist):
            if b.n == 0:
                scores[i] = float(self.sil_.intercept_ or 0.0)
            else:
                scores[i] = float(np.max(self.sil_.decision_function(b.X)))
        return scores

    def predict(self, bags: Sequence[Bag] | BagDataset | Sequence[np.ndarray]) -> npt.NDArray[np.float64]:
        return (self.decision_function(bags) >= 0.0).astype(float)

    def score(self, bags, y_true) -> float:
        y_pred = self.predict(bags)
        if isinstance(bags, BagDataset):
            y_true_arr = np.asarray([b.y for b in bags.bags], dtype=float)
        elif len(bags) and isinstance(bags[0], Bag):  # type: ignore[index]
            y_true_arr = np.asarray([b.y for b in bags], dtype=float)
        else:
            y_true_arr = np.asarray(y_true, dtype=float)
        return float((y_pred == y_true_arr).mean())
