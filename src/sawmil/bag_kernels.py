# sparse_mil/bag_kernels.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from .bag import Bag
from .kernels import BaseKernel, Linear

# ---------- Normalizer Utils
_NormalizerName = Literal["none", "average", "featurespace"]

# ---------- Base Multiple-instance Kernel

def _bag_slices(bags):
    starts = np.fromiter((0,), int)
    for b in bags:
        starts = np.append(starts, starts[-1] + b.n)
    return starts  # length = n_bags+1

def _weights_for_normalizer(bags, normalizer: str):
    out, lens = [], []
    for b in bags:
        n = b.n
        lens.append(n)
        if n == 0:
            out.append(np.zeros((0,), float))
        elif normalizer == "average":
            out.append(np.ones(n, float))        # “sum” weights; divide by counts later
        else:
            out.append(np.full(n, 1.0 / n, float))  # “mean” weights
    w = np.concatenate(out) if out else np.zeros((0,), float)
    return w, np.array(lens, int)

def _segment_reduce_rows(K, w_rows, starts):
    # Multiply each row by its weight, then sum segments -> (n_bags_X, n_inst_Y)
    KR = K * w_rows[:, None]
    return np.add.reduceat(KR, starts[:-1], axis=0)

def _segment_reduce_cols(M, w_cols, starts):
    # Multiply each column by its weight, then sum segments -> (n_bags_X, n_bags_Y)
    MC = M * w_cols[None, :]
    return np.add.reduceat(MC, starts[:-1], axis=1)


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
    A simple, readable bag kernel built from an instance kernel k(·,·).

    Definition
    ----------
    For two bags Bi and Bj with instance sets Xi and Xj, let w_i and w_j be
    per-instance weights (see below). The kernel is

        K(Bi, Bj) = [(w_i^T K(Xi, Xj) w_j)]^p / (norm(Bi) * norm(Bj)),

    where K(Xi, Xj) is the instance-level Gram submatrix produced by
    `inst_kernel` and p>=1 is an optional element-wise exponent (default 1.0).

    Weights and normalizers
    -----------------------
    - normalizer="none":
        w is the mean weight (1/n per instance). No additional normalization;
        K(Bi,Bj) equals the average pairwise instance kernel value.
    - normalizer="average" (default):
        w is the sum weight (1 per instance). The numerator becomes the sum
        over all instance pairs, and dividing by norm(Bi)=|Bi| and
        norm(Bj)=|Bj| yields the same average over pairs as above, but kept in
        a form that generalizes to non-linear feature-space norms.
    - normalizer="featurespace":
        w is the mean weight (1/n), and
        norm(B) = sqrt(w^T K(X, X) w), i.e. the feature-space norm of the
        (weighted) mean embedding. For a Linear instance kernel this is just
        the Euclidean norm of the mean vector.

    Notes
    -----
    - For Linear instance kernels and p=1, this recovers the dot-product of
      (possibly normalized) bag means.
    - When applying the exponent p>1 we clamp negatives to 0 before powering
      to help preserve PSD-like behaviour for numerical stability.
    """
    inst_kernel: BaseKernel
    normalizer: _NormalizerName = "average"
    p: float = 1.0


    def fit(self, bags: List[Bag]) -> "WeightedMeanBagKernel":
        """Fit the underlying instance kernel if it needs data-dependent defaults.

        We just peek at the first non-empty bag to infer dimensionality for
        kernels like RBF that auto-derive parameters (e.g., gamma).
        """
        # If the instance kernel needs defaults (e.g. gamma), fit it on a few instances.
        # We can use the first bag to infer dimensionality.
        for b in bags:
            if b.n > 0:
                self.inst_kernel.fit(b.X)
                break
        return self

    def __callY__(self, bags_X: list[Bag], bags_Y: list[Bag]) -> npt.NDArray[np.float64]:
        nX, nY = len(bags_X), len(bags_Y)

        X_stack = np.vstack([b.X for b in bags_X if b.n])
        Y_stack = X_stack if (bags_X is bags_Y) else np.vstack([b.X for b in bags_Y if b.n])

        # Build K_inst once
        K_inst = self.inst_kernel(X_stack, Y_stack)  # (N_X, N_Y)

        # Flat weights and segment boundaries
        wX_flat, lensX = _weights_for_normalizer(bags_X, self.normalizer)
        wY_flat, lensY = _weights_for_normalizer(bags_Y, self.normalizer)
        startsX = np.concatenate(([0], np.cumsum(lensX)))
        startsY = np.concatenate(([0], np.cumsum(lensY)))

        # Two-stage aggregation (no SciPy sparse)
        SXK   = _segment_reduce_rows(K_inst, wX_flat, startsX)   # (n_bags_X, N_Y)
        K_bag = _segment_reduce_cols(SXK, wY_flat, startsY)      # (n_bags_X, n_bags_Y)

        # ---- norms
        if self.normalizer == "none":
            norms_X = np.ones(nX, dtype=float)
            norms_Y = norms_X if (bags_X is bags_Y) else np.ones(nY, dtype=float)
        elif self.normalizer == "average":
            norms_X = np.array([max(b.n, 1) for b in bags_X], dtype=float)
            norms_Y = norms_X if (bags_X is bags_Y) else np.array([max(b.n, 1) for b in bags_Y], dtype=float)
            
                # ---- apply exponent p and normalization
        if self.p != 1.0:
            np.maximum(K_bag, 0.0, out=K_bag)  # keep PSD-ish
            K_bag = np.power(K_bag, self.p, dtype=float)

        denom = np.outer(norms_X, norms_Y)
        np.divide(K_bag, denom, out=K_bag, where=(denom > 0))

        return K_bag



    def __call__(self, bags_X: list[Bag], bags_Y: list[Bag]) -> npt.NDArray[np.float64]:
        """Compute the bag-by-bag Gram matrix.

        Args:
            bags_X: list of Bag objects (left side)
            bags_Y: list of Bag objects (right side)

        Returns:
            A (len(bags_X) x len(bags_Y)) matrix K where K[i,j] is the kernel
            value between bags_X[i] and bags_Y[j] under the configuration
            described in the class docstring.
        """


        nX, nY = len(bags_X), len(bags_Y)

        # Handle empty bag lists quickly
        if nX == 0 or nY == 0:
            return np.zeros((nX, nY), dtype=float)

        # ---- build stacked instance matrices
        d = bags_X[0].d if nX else 0

        X_stack = np.vstack([b.X if b.n else np.zeros((0, d), float) for b in bags_X])
        Y_stack = X_stack if (bags_X is bags_Y) else \
                np.vstack([b.X if b.n else np.zeros((0, d), float) for b in bags_Y])

        # ---- per-instance weights per bag
        # 
        def w_vec(b: Bag) -> np.ndarray:
            """Weight vector for a bag b.
            Returns:
                A 1D array of shape (b.n,) with the weight for each instance.
                
            Notes
            -----
            - "average": use 1 per instance (sum); divide by |Bi||Bj| later.
            - else ("none" or "featurespace"): use mean weights (1/n per instance).
            """
            if b.n == 0:
                return np.zeros((0,), dtype=float)
            if self.normalizer == "average":
                return np.full(b.n, 1.0 / b.n, dtype=float)  # mean
            return np.ones(b.n, dtype=float)     # sum= n



        wX = [w_vec(b) for b in bags_X]
        wY = wX if (bags_X is bags_Y) else [w_vec(b) for b in bags_Y]

        # ---- sparse selection/weighting matrices Sx, Sy
        # Shape: (n_bags, n_instances). Row i selects/weights instances of bag i.
        def make_S(weights_list: list[np.ndarray]) -> sp.csr_matrix:
            """ Matrix of shape (n_bags, n_instances) with weights for each instance in each bag.
            Tracks which instance belongs to which bag, and applies the per-instance weights.
            """
            rows, cols, data = [], [], []
            offset = 0
            for i, w in enumerate(weights_list):
                if w.size:
                    j = np.arange(w.size) + offset
                    rows.extend([i] * w.size)
                    cols.extend(j.tolist())
                    data.extend(w.tolist())
                offset += w.size
            return sp.csr_matrix((data, (rows, cols)), shape=(len(weights_list), offset))

        Sx = make_S(wX)
        Sy = Sx if (bags_X is bags_Y) else make_S(wY)

        # ---- one big instance-kernel call, then reduce by Sx/Sy
        # K_inst has shape (#instances in X) x (#instances in Y)
        K_inst = self.inst_kernel(X_stack, Y_stack)

        # ---- aggregate to bag-bag
        # K_bag[i,j] = w_i^T K_inst(block_i, block_j) w_j  <=>  Sx @ K_inst @ Sy.T
        K_bag = (Sx @ K_inst) @ Sy.T
        K_bag = np.asarray(K_bag, dtype=float)

        # ---- norms
        if self.normalizer == "none":
            norms_X = np.ones(nX, dtype=float)
            norms_Y = norms_X if (bags_X is bags_Y) else np.ones(nY, dtype=float)
        elif self.normalizer == "average":
            norms_X = np.array([max(b.n, 1) for b in bags_X], dtype=float)
            norms_Y = norms_X if (bags_X is bags_Y) else np.array([max(b.n, 1) for b in bags_Y], dtype=float)
        else:  # "featurespace"
            if isinstance(self.inst_kernel, Linear):
                # For linear kernels, the feature map is the identity, so the
                # mean embedding is simply the (weighted) mean vector.
                # means: (n_bags, d) = Sx @ X_stack
                means_X = Sx @ X_stack
                norms_X = np.linalg.norm(means_X, axis=1).clip(min=1e-12)
                if bags_X is bags_Y:
                    norms_Y = norms_X
                else:
                    means_Y = Sy @ Y_stack
                    norms_Y = np.linalg.norm(means_Y, axis=1).clip(min=1e-12)
            else:
                # general featurespace norm via self-gram reduction
                # K_self_X = Sx @ (inst_kernel(X,X)) @ Sx.T
                K_self_X = (Sx @ (self.inst_kernel(X_stack, X_stack))) @ Sx.T
                norms_X = np.sqrt(np.maximum(np.diag(K_self_X), 1e-12))
                if bags_X is bags_Y:
                    norms_Y = norms_X
                else:
                    K_self_Y = (Sy @ (self.inst_kernel(Y_stack, Y_stack))) @ Sy.T
                    norms_Y = np.sqrt(np.maximum(np.diag(K_self_Y), 1e-12))

        # ---- apply exponent p and normalization
        if self.p != 1.0:
            np.maximum(K_bag, 0.0, out=K_bag)  # keep PSD-ish
            K_bag = np.power(K_bag, self.p, dtype=float)

        denom = np.outer(norms_X, norms_Y)
        np.divide(K_bag, denom, out=K_bag, where=(denom > 0))

        return K_bag


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
) -> WeightedMeanBagKernel:
    '''Helper to create a WeightedMeanBagKernel with the given instance kernel and parameters.
    Args:
      inst_kernel: BaseKernel
        The instance-level kernel to use (will be fitted on first bag with instances).
      normalizer: {"none", "average", "featurespace"}
        The bag kernel normalizer (default is "none").
      p: float
        Exponent for the weighted mean kernel (default is 1.0, i.e. no exponentiation).
    '''
    return WeightedMeanBagKernel(
        inst_kernel=inst_kernel,
        normalizer=normalizer,
        p=p,
    )


__all__ = [
    "WeightedMeanBagKernel",
    "PrecomputedBagKernel",
    "make_bag_kernel",
    "BaseBagKernel"
]
