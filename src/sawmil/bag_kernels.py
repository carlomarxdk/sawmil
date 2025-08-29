# sparse_mil/bag_kernels.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal
import numpy as np
import numpy.typing as npt

from .bag import Bag
from .kernels import BaseKernel, Linear

# ---------- Normalizer Utils 
_NormalizerName = Literal["none", "average", "featurespace"]

# ---------- Base Multiple-instance Kernel
class BaseBagKernel(ABC):
    '''Base class for the Multiple-instance (bags) kernel'''
    def fit(self, bags: List[Bag]) -> "BaseBagKernel":
        return self
    @abstractmethod
    def __call__(self, bags_X: List[Bag], bags_Y: List[Bag]) -> npt.NDArray[np.float64]:
        ...

def _effective_count(b: Bag) -> float:
    '''Counts the effective instances in a bag (relevant only if classifier or kernel uses the intra_bag_labels)'''
    s = float(b.mask.sum())
    return s if s > 0.0 else max(1.0, float(b.n))
    
@dataclass
class WeightedMeanBagKernel(BaseBagKernel):
    """
    K(Bi,Bj) = [ (w_i^T k(Bi,Bj) w_j) ] ** p / ( norm(Bi) * norm(Bj) ),
      where w_i, w_j are instance weights normalized to sum=1
      (fallback to uniform if a bag’s mask sums to 0).

    normalizer: (default is 'average')
      - "none"       -> norm(B)=1
      - "average"    -> norm(B)=sum(mask) (fallback to bag size)
      - "featurespace" -> norm(B)=sqrt(w^T k(X,X) w)
                          (fast for Linear via ||weighted_mean||)

    By default (use_intra_labels=False):
      - w_i are UNIFORM weights (1/n_i), i.e., intra labels are IGNORED.
      - normalizer defaults to "none" (no extra scaling).
    """
    inst_kernel: BaseKernel
    normalizer: _NormalizerName = "average"
    p: float = 1.0
    use_intra_labels: bool = False
    fast_linear: bool = True

    def fit(self, bags: List[Bag]) -> "WeightedMeanBagKernel":
        # If the instance kernel needs defaults (e.g. gamma), fit it on a few instances.
        # We can use the first bag to infer dimensionality.
        for b in bags:
            if b.n > 0:
                self.inst_kernel.fit(b.X)
                break
        return self

    # ---- helpers ----
    def _weights(self, b: Bag) -> npt.NDArray[np.float64]:
        if b.n == 0:
            return np.zeros((0,), dtype=float)

        if not self.use_intra_labels:
            # 👇 KEY CHANGE:
            # If "average" normalizer is requested, use UNNORMALIZED weights (all ones)
            # so the numerator is a SUM; the denominator will divide by counts -> net MEAN.
            if self.normalizer == "average":
                return np.ones(b.n, dtype=float)
            # otherwise, keep the plain mean
            return np.full(b.n, 1.0 / b.n, dtype=float)

        # use intra-bag mask (normalized to sum=1; fallback to uniform)
        w = b.mask.astype(float)
        s = float(w.sum())
        if s <= 0.0:
            return np.full(b.n, 1.0 / b.n, dtype=float)
        return w / s

    def _norms(self, bags: List[Bag]) -> npt.NDArray[np.float64]:
        if self.normalizer == "none":
            return np.ones(len(bags), dtype=float)

        if self.normalizer == "average":
            # 👇 CONSISTENT with _weights above:
            # - no intra labels: denominator = bag size
            # - with intra labels: denominator = effective positive count
            if not self.use_intra_labels:
                return np.array([max(b.n, 1) for b in bags], dtype=float)
            else:
                return np.array([max(float(b.mask.sum()), 1.0) for b in bags], dtype=float)

        # featurespace
        if self.fast_linear and isinstance(self.inst_kernel, Linear):
            means = np.stack(
                [self._weighted_mean(b) if b.n else np.zeros((bags[0].d,), dtype=float)
                 for b in bags],
                axis=0,
            )
            return np.linalg.norm(means, axis=1).clip(min=1e-12)
        norms = np.empty(len(bags), dtype=float)
        for i, b in enumerate(bags):
            if b.n == 0:
                norms[i] = 1.0
                continue
            w = self._weights(b)
            G = self.inst_kernel(b.X, b.X)
            val = float(w @ G @ w)
            norms[i] = np.sqrt(max(val, 1e-12))
        return norms

    def _pair_value(self, Bi: Bag, Bj: Bag) -> float:
        if Bi.n == 0 or Bj.n == 0:
            return 0.0
        wi = self._weights(Bi)
        wj = self._weights(Bj)
        if self.fast_linear and isinstance(self.inst_kernel, Linear):
            mi = (wi[None, :] @ Bi.X).ravel()
            mj = (wj[None, :] @ Bj.X).ravel()
            val = float(mi @ mj)
        else:
            G = self.inst_kernel(Bi.X, Bj.X)
            val = float(wi @ G @ wj)
        if self.p != 1.0:
            val = float(np.power(max(val, 0.0), self.p))
        return val

    def __call__(self, bags_X: List[Bag], bags_Y: List[Bag]) -> npt.NDArray[np.float64]:
        nX, nY = len(bags_X), len(bags_Y)
        K = np.empty((nX, nY), dtype=float)
        same = bags_X is bags_Y
        norms_X = self._norms(bags_X)
        norms_Y = norms_X if same else self._norms(bags_Y)
        for i, Bi in enumerate(bags_X):
            j0 = i if same else 0
            for j in range(j0, nY):
                val = self._pair_value(Bi, bags_Y[j])
                denom = norms_X[i] * norms_Y[j]
                kij = val / denom if denom > 0.0 else 0.0
                K[i, j] = kij
                if same and j != i:
                    K[j, i] = kij
        return K
    
# ---------- Precomputed bag kernel ----------
@dataclass
class PrecomputedBagKernel(BaseBagKernel):
    K: npt.NDArray[np.float64]
    def __call__(self, bags_X: List[Bag], bags_Y: List[Bag]) -> npt.NDArray[np.float64]:
        # Caller must pass consistent ordering to match K
        return np.asarray(self.K, dtype=float)

# ---------- Simple factory ----------
def make_bag_kernel(
    inst_kernel: BaseKernel,
    *,
    normalizer: _NormalizerName = "none",
    p: float = 1.0,
    use_intra_labels: bool = False,
    fast_linear: bool = True,
) -> WeightedMeanBagKernel:
    return WeightedMeanBagKernel(
        inst_kernel=inst_kernel,
        normalizer=normalizer,
        p=p,
        use_intra_labels=use_intra_labels,
        fast_linear=fast_linear,
    )


__all__ = [
    "WeightedMeanBagKernel",
    "PrecomputedBagKernel",
    "make_bag_kernel",
    "BaseBagKernel"
]