# sparse_mil/smil.py
from __future__ import annotations
from typing import Optional, Sequence, List
import numpy as np
import numpy.typing as npt

from .nsk import NSK
from .bag import Bag, BagDataset
from .bag_kernels import WeightedMeanBagKernel
from .kernels import Linear
from .quadprog import quadprog_gurobi


class sMIL(NSK):
    """
    Sparse MIL (Bunescu & Mooney, 2007) implemented on top of NSK.

    Training set:
      - Every negative *instance* becomes its own 1-instance bag (label -1).
      - Every positive bag stays a bag (label +1).

    Dual tweaks:
      - Linear term for positive bags: f_j = 2/|B_j| - 1
      - Box constraints: iC for negatives, bC for positives (scaled if scale_C)

    Notes:
      - By default we ignore intra-bag labels (uniform instance weights).
      - Use your NSK's bag kernel; mean aggregator with normalizer="none" is a good default.
    """

    @staticmethod
    def _coerce_bags_and_labels(
        bags: Sequence[Bag] | BagDataset | Sequence[np.ndarray],
        y: Optional[npt.NDArray[np.float64]] = None
    ) -> tuple[List[Bag], npt.NDArray[np.float64]]:
        # Reuse NSK’s accepted inputs; prefer Bag.y when available
        if isinstance(bags, BagDataset):
            bl = list(bags.bags)
            ya = np.asarray([b.y for b in bl], dtype=float)
            return bl, ya
        if len(bags) > 0 and isinstance(bags[0], Bag):  # type: ignore[index]
            bl = list(bags)  # type: ignore[assignment]
            ya = np.asarray([b.y for b in bl], dtype=float) if y is None else np.asarray(y, dtype=float).ravel()
            if y is not None and ya.shape[0] != len(bl):
                raise ValueError("Length of y must equal number of bags.")
            return bl, ya
        if y is None:
            raise ValueError("When passing raw arrays for bags, you must also pass y.")
        bl = [Bag(X=np.asarray(b, dtype=float), y=float(lbl)) for b, lbl in zip(bags, y)]
        return bl, np.asarray(y, dtype=float).ravel()

    @staticmethod
    def _build_init_training(bags: List[Bag]) -> tuple[List[Bag], npt.NDArray[np.float64], int, int]:
        # negative instances -> singleton bags
        negative_instances: List[Bag] = [
            Bag(X=b.X[j:j+1, :], y=-1.0)
            for b in bags if float(b.y) <= 0.0
            for j in range(b.n)
        ]
        # positive bags unchanged
        positive_bags: List[Bag] = [b for b in bags if float(b.y) > 0.0]

        train_bags = negative_instances + positive_bags
        S_n = len(negative_instances)      # # negative instances
        B_p = len(positive_bags)           # # positive bags
        y_train = np.hstack([-np.ones(S_n), np.ones(B_p)])
        return train_bags, y_train, S_n, B_p

    def fit(self, bags: Sequence[Bag] | BagDataset | Sequence[np.ndarray],
            y: Optional[npt.NDArray[np.float64]] = None) -> "sMIL":
        # 1) coerce inputs and build sMIL training set
        init_bags, _ = self._coerce_bags_and_labels(bags, y)
        train_bags, y_train, S_n, B_p = self._build_init_training(init_bags)
        if not train_bags:
            raise ValueError("No training data after sMIL transformation.")
        self.bags_ = train_bags

        # map to {-1,+1} and store classes
        classes = np.unique(y_train)
        if classes.size != 2:
            raise ValueError("Binary classification only.")
        self.classes_ = classes.astype(float)
        Y = np.where(y_train == classes[0], -1.0, 1.0)
        self.y_ = Y

        # 2) bag kernel Gram
        bk = self._ensure_bag_kernel()
        bk.fit(train_bags)
        K = bk(train_bags, train_bags)                     # (n, n)

        # 3) QP pieces
        H = (Y[:, None] * Y[None, :]) * K
        H = 0.5 * (H + H.T)
        n = len(train_bags)
        f = -np.ones(n, dtype=float)
        sizes = np.array([b.n for b in train_bags[S_n:]], dtype=float)
        f[S_n:] = (2.0 / np.maximum(sizes, 1.0)) - 1.0

        Aeq = Y.reshape(1, -1)
        beq = np.array([0.0], dtype=float)

        # per-variable box constraints
        if self.scale_C:
            iC = float(self.C) / max(S_n, 1)
            bC = float(self.C) / max(B_p, 1)
        else:
            iC = float(self.C)
            bC = float(self.C)
        lb = np.zeros(n, dtype=float)
        ub = np.concatenate([np.full(S_n, iC), np.full(B_p, bC)]).astype(float)

        # 4) solve
        alpha, _ = quadprog_gurobi(H, f, Aeq, beq, lb, ub, verbose=self.verbose)
        self.alpha_ = alpha

        # 5) SVs + intercept (dual)
        sv_mask = alpha > self.tol
        self.support_ = np.flatnonzero(sv_mask).astype(int)
        self.support_vectors_ = [train_bags[i] for i in self.support_]
        self.dual_coef_ = (alpha[sv_mask] * Y[sv_mask]).reshape(1, -1)

        caps = np.concatenate([np.full(S_n, iC), np.full(B_p, bC)]).astype(float)
        on_margin = (alpha > self.tol) & (alpha < (caps - self.tol))
        if not np.any(on_margin):
            on_margin = sv_mask
        b_vals = Y[on_margin] - (alpha * Y) @ K[:, on_margin]
        self.intercept_ = float(np.mean(b_vals)) if b_vals.size else 0.0

        # 6) optional primal recovery (linearizable case: Linear + WeightedMean + p==1)
        if isinstance(bk, WeightedMeanBagKernel) and isinstance(bk.inst_kernel, Linear) and abs(bk.p - 1.0) < 1e-12:
            # φ(B) consistent with your NSK (uniform weights, chosen normalizer)
            Z = np.stack([self._phi(b, normalizer=bk.normalizer) for b in train_bags], axis=0)  # (n, d)
            self.coef_ = (self.alpha_ * self.y_) @ Z
            use = on_margin if np.any(on_margin) else sv_mask
            if np.any(use):
                self.intercept_ = float(np.mean(self.y_[use] - Z[use] @ self.coef_))
        else:
            self.coef_ = None  # keep dual intercept

        return self
