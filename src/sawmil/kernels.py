# sparse_mil/kernels.py
# Kernel Implementation for the single-instance cases
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union
import numpy as np
import numpy.typing as npt
import logging

log = logging.getLogger("sparse_mil.kernels")


# ---------------- Utilities 
def _sqeuclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xn = (X * X).sum(1)[:, None]
    Yn = (Y * Y).sum(1)[None, :]
    return Xn + Yn - 2.0 * (X @ Y.T)

# ---------------- Base class 
class BaseKernel(ABC):
    """Minimal kernel interface for the single-instance kernels: fit (optional) + __call__."""
    def fit(self, X: npt.NDArray[np.float64]) -> "BaseKernel":
        return self
    @abstractmethod
    def __call__(self, X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ...

# ----------------  Single-instance kernels 
@dataclass
class Linear(BaseKernel):
    def fit(self, X: npt.NDArray[np.float64]) -> "Linear":
        log.warning("Linear kernel has no parameters to fit.")
        return self
    def __call__(self, X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]) -> np.ndarray:
        return X @ Y.T

@dataclass
class RBF(BaseKernel):
    gamma: Optional[float] = None  # if None, set to 1/d in fit()
    def fit(self, X: npt.NDArray[np.float64]) -> "RBF":
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        return self
    def __call__(self, X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]) -> np.ndarray:
        assert self.gamma is not None, "Call .fit(X) or set gamma."
        return np.exp(-self.gamma * _sqeuclidean(X, Y))

@dataclass
class Polynomial(BaseKernel):
    degree: int = 3
    gamma: Optional[float] = None
    coef0: float = 0.0
    def fit(self, X: npt.NDArray[np.float64]) -> "Polynomial":
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        return self
    def __call__(self, X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]) -> np.ndarray:
        assert self.gamma is not None
        return (self.gamma * (X @ Y.T) + self.coef0) ** self.degree

@dataclass
class Sigmoid(BaseKernel):
    gamma: Optional[float] = None
    coef0: float = 0.0
    def fit(self, X: npt.NDArray[np.float64]) -> "Sigmoid":
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        return self
    def __call__(self, X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]) -> np.ndarray:
        assert self.gamma is not None
        return np.tanh(self.gamma * (X @ Y.T) + self.coef0)

@dataclass
class Precomputed(BaseKernel):
    """Use when a Gram matrix is already built; ignores X,Y and returns K (shape checked by caller)."""
    K: np.ndarray
    def fit(self, X: npt.NDArray[np.float64]) -> "Precomputed":
        log.warning("Precomputed kernel has no parameters to fit.")
        return self
    def __call__(self, X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]) -> np.ndarray:
        return self.K

# ---------------- Combinators 
@dataclass
class Scale(BaseKernel):
    a: float
    k: BaseKernel
    def fit(self, X): self.k.fit(X); return self
    def __call__(self, X, Y): return self.a * self.k(X, Y)

@dataclass
class Sum(BaseKernel):
    k1: BaseKernel
    k2: BaseKernel
    def fit(self, X): self.k1.fit(X); self.k2.fit(X); return self
    def __call__(self, X, Y): return self.k1(X, Y) + self.k2(X, Y)

@dataclass
class Product(BaseKernel):
    k1: BaseKernel
    k2: BaseKernel
    def fit(self, X): self.k1.fit(X); self.k2.fit(X); return self
    def __call__(self, X, Y): return self.k1(X, Y) * self.k2(X, Y)

@dataclass
class Normalize(BaseKernel):
    """Cosine-style normalization: Kxy / sqrt(Kxx * Kyy)."""
    k: BaseKernel
    eps: float = 1e-12
    def fit(self, X): self.k.fit(X); return self
    def __call__(self, X, Y):
        Kxy = self.k(X, Y)
        Kxx = np.sqrt(np.maximum(np.diag(self.k(X, X)), self.eps))[:, None]
        Kyy = np.sqrt(np.maximum(np.diag(self.k(Y, Y)), self.eps))[None, :]
        return Kxy / (Kxx * Kyy + self.eps)

# ---------------- Registry & Resolver 
# Used for the type checking
KernelType = Union[
    str,                                     # "linear", "rbf", "poly", "sigmoid", "precomputed"
    BaseKernel,                               # already-constructed kernel
    Callable[[np.ndarray, np.ndarray], np.ndarray],  # raw callable k(X, Y)
]

def get_kernel(spec: KernelType, **kwargs) -> BaseKernel:
    """
    Normalize various 'kernel=' inputs into a BaseKernel object.
    kwargs are used only when spec is a string (e.g., gamma, degree, coef0, K).
    """
    # already an object?
    if isinstance(spec, BaseKernel):
        return spec

    # plain callable? wrap so it behaves like a BaseKernel
    if callable(spec) and not isinstance(spec, str):
        class _CallableKernel(BaseKernel):
            def __call__(self, X, Y): return np.asarray(spec(X, Y), dtype=float)
        return _CallableKernel()

    # by name
    name = str(spec).lower()
    if name == "precomputed":
        if "K" not in kwargs:
            raise ValueError("get_kernel('precomputed', K=...) requires the Gram matrix K.")
        return Precomputed(np.asarray(kwargs["K"], dtype=float))

    registry: Dict[str, BaseKernel] = {
        "linear":  Linear(),
        "rbf":     RBF(kwargs.get("gamma")),
        "poly":    Polynomial(kwargs.get("degree", 3), kwargs.get("gamma"), kwargs.get("coef0", 0.0)),
        "sigmoid": Sigmoid(kwargs.get("gamma"), kwargs.get("coef0", 0.0)),
    }
    if name in registry:
        return registry[name]

    raise ValueError(f"Unknown kernel spec: {spec!r}")

__all__ = [
    "BaseKernel",
    "Linear", "RBF", "Polynomial", "Sigmoid", "Precomputed",
    "Scale", "Sum", "Product", "Normalize",
    "KernelType", "get_kernel",
]