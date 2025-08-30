# utils/make_complex_bags.py
from __future__ import annotations
from typing import Iterable, Sequence, Tuple
import numpy as np
from numpy.random import Generator
from dataclasses import dataclass

try:
    from ..bag import Bag, BagDataset  # adjust import to your paths
except ImportError:
    from sawmil.bag import Bag, BagDataset  # fallback import

@dataclass
class GaussianComp:
    mu: np.ndarray           # (d,)
    cov: np.ndarray          # (d,d)

def _rand_rot_cov(rng: Generator, d: int, scale: float = 1.0, anisotropy: float = 0.5) -> np.ndarray:
    """
    Random SPD covariance: R diag(s^2) R^T
    - 'scale' controls overall variance
    - 'anisotropy' in [0,1]: 0 -> spherical, 1 -> very elongated
    """
    # random orthogonal via QR
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    # log-spaced eigenvalues
    max_eig = (scale ** 2)
    min_eig = max_eig * (1.0 - anisotropy)
    evals = np.linspace(max_eig, min_eig, d)
    return Q @ np.diag(evals) @ Q.T

def _components_from_centers(rng: Generator, centers: Sequence[Sequence[float]],
                             scales: Sequence[Tuple[float, float]]) -> list[GaussianComp]:
    """
    Build Gaussian components given (mu) centers and (scale, anisotropy) per component.
    """
    comps: list[GaussianComp] = []
    d = len(centers[0])
    for mu, (scale, aniso) in zip(centers, scales):
        cov = _rand_rot_cov(rng, d=d, scale=scale, anisotropy=aniso)
        comps.append(GaussianComp(mu=np.asarray(mu, dtype=float), cov=cov))
    return comps

def _sample_comp(rng: Generator, comp: GaussianComp, n: int) -> np.ndarray:
    return rng.multivariate_normal(mean=comp.mu, cov=comp.cov, size=n)

def generate_dummy_bags(
    *,
    n_pos: int = 100,
    n_neg: int = 60,
    inst_per_bag: Tuple[int, int] = (4, 12),     # bag size ~ Uniform[min,max]
    d: int = 2,
    # Positive & negative class mixtures
    pos_centers: Sequence[Sequence[float]] = ((+2.0, +1.0), (+4.0, +3.0)),
    neg_centers: Sequence[Sequence[float]] = ((-1.5, -1.0), (-3.0, +0.5)),
    pos_scales: Sequence[Tuple[float, float]] = ((2.0, 0.6), (1.2, 0.8)),  # (scale, anisotropy)
    neg_scales: Sequence[Tuple[float, float]] = ((1.5, 0.5), (2.5, 0.9)),
    # Intra-bag positives (only matter for positive bags)
    pos_intra_rate: Tuple[float, float] = (0.3, 0.8),  # per-bag fraction range for intra==1
    ensure_pos_in_every_pos_bag: bool = True,
    # Cross-contamination
    neg_pos_noise_rate: Tuple[float, float] = (0.00, 0.05),  # fraction of pos-like instances inside *negative* bags
    pos_neg_noise_rate: Tuple[float, float] = (0.00, 0.20),  # fraction of neg-like instances inside *positive* bags
    # Outliers sprinkled everywhere
    outlier_rate: float = 0.01,
    outlier_scale: float = 10.0,
    random_state: int = 0,
) -> BagDataset:
    """Generate a synthetic MIL dataset with mixed Gaussian components.

    Returns:
        BagDataset:
            - Positive bags (y=1) mix pos-like instances (intra=1) and distractors (intra=0).
            - Negative bags (y=0) may include a small fraction of pos-like contamination; intra
              labels default to ones in Bag, but the bag label remains 0.
    """
    rng = np.random.default_rng(random_state)
    assert inst_per_bag[0] >= 1 and inst_per_bag[1] >= inst_per_bag[0]
    assert len(pos_centers) == len(pos_scales)
    assert len(neg_centers) == len(neg_scales)
    assert d == len(pos_centers[0]) == len(neg_centers[0])

    pos_comps = _components_from_centers(rng, pos_centers, pos_scales)
    neg_comps = _components_from_centers(rng, neg_centers, neg_scales)

    def sample_from_mix(n: int, comps: list[GaussianComp]) -> np.ndarray:
        # uniform mixture over components
        ks = rng.integers(0, len(comps), size=n)
        X = np.empty((n, d), dtype=float)
        for k in range(len(comps)):
            idx = np.where(ks == k)[0]
            if idx.size:
                X[idx] = _sample_comp(rng, comps[k], idx.size)
        return X

    def add_outliers(X: np.ndarray, frac: float) -> None:
        if frac <= 0: return
        m = X.shape[0]
        n_out = int(round(frac * m))
        if n_out > 0:
            # Heavy-tailed-ish outliers
            X[:n_out] += rng.normal(scale=outlier_scale, size=(n_out, d))

    bags: list[Bag] = []

    # ---- Positive bags ----
    for _ in range(n_pos):
        m = int(rng.integers(inst_per_bag[0], inst_per_bag[1] + 1))
        # how many intra==1?
        r_pos = rng.uniform(*pos_intra_rate)
        n_pos_inst = int(round(r_pos * m))
        if ensure_pos_in_every_pos_bag:
            n_pos_inst = max(1, n_pos_inst)
        n_neg_like = m - n_pos_inst

        # fraction of neg-like distractors inside positive bag
        r_distr = rng.uniform(*pos_neg_noise_rate)
        n_neg_like = max(0, n_neg_like)  # in case n_pos_inst == m
        # sample positives from pos-mixture
        X_pos = sample_from_mix(n_pos_inst, pos_comps) if n_pos_inst > 0 else np.zeros((0, d))
        # sample distractors from neg-mixture
        X_distr = sample_from_mix(n_neg_like, neg_comps) if n_neg_like > 0 else np.zeros((0, d))
        X = np.vstack([X_pos, X_distr]) if X_distr.size else X_pos

        # optional: sprinkle outliers
        add_outliers(X, outlier_rate)

        # intra labels: 1 for pos-like, 0 for distractors
        intra = np.concatenate([np.ones(n_pos_inst), np.zeros(n_neg_like)]).astype(float)
        # shuffle instances within the bag
        perm = rng.permutation(X.shape[0])
        X = X[perm]
        intra = intra[perm]

        bags.append(Bag(X=X, y=1.0, intra_bag_label=intra))

    # ---- Negative bags ----
    for _ in range(n_neg):
        m = int(rng.integers(inst_per_bag[0], inst_per_bag[1] + 1))
        # proportion of pos-like contamination (still y=0 at bag level)
        r_noise = rng.uniform(*neg_pos_noise_rate)
        n_pos_like = int(round(r_noise * m))
        n_neg_core = m - n_pos_like

        X_neg_core = sample_from_mix(n_neg_core, neg_comps) if n_neg_core > 0 else np.zeros((0, d))
        X_pos_noise = sample_from_mix(n_pos_like, pos_comps) if n_pos_like > 0 else np.zeros((0, d))
        X = np.vstack([X_neg_core, X_pos_noise]) if X_pos_noise.size else X_neg_core

        add_outliers(X, outlier_rate)

        # For negative bags, intra labels default to ones in your Bag class—but
        # they are ignored by your negative_instances(); we can still pass all ones.
        intra = np.ones(X.shape[0], dtype=float)
        perm = rng.permutation(X.shape[0])
        X = X[perm]; intra = intra[perm]

        bags.append(Bag(X=X, y=0.0, intra_bag_label=intra))

    return BagDataset(bags)
