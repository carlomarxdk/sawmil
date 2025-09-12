import numpy as np
import pytest

try:
    from sawmil.kernels import (
        Linear,
        RBF,
        Polynomial,
        Sigmoid,
        Precomputed,
        Scale,
        Sum,
        Product,
        Normalize,
        get_kernel,
    )
except Exception:  # pragma: no cover
    from ..src.sawmil.kernels import (
        Linear,
        RBF,
        Polynomial,
        Sigmoid,
        Precomputed,
        Scale,
        Sum,
        Product,
        Normalize,
        get_kernel,
    )


def test_linear_kernel_returns_dot_product():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    Y = np.array([[5, 6], [7, 8]], dtype=float)
    k = Linear()
    assert np.allclose(k(X, Y), X @ Y.T)


def test_rbf_kernel_fit_sets_gamma_and_computes():
    X = np.array([[1, 0], [0, 1]], dtype=float)
    k = RBF()
    k.fit(X)
    assert k.gamma == pytest.approx(0.5)
    K = k(X, X)
    expected = np.exp(-0.5 * np.array([[0.0, 2.0], [2.0, 0.0]]))
    assert np.allclose(K, expected)


def test_polynomial_and_sigmoid_kernels():
    X = np.array([[1, 2]], dtype=float)
    Y = np.array([[3, 4]], dtype=float)

    poly = Polynomial(degree=2, gamma=0.5, coef0=1.0)
    K_poly = poly(X, Y)
    expected_poly = (0.5 * (X @ Y.T) + 1.0) ** 2
    assert np.allclose(K_poly, expected_poly)

    sig = Sigmoid(gamma=0.5, coef0=-1.0)
    K_sig = sig(X, Y)
    expected_sig = np.tanh(0.5 * (X @ Y.T) - 1.0)
    assert np.allclose(K_sig, expected_sig)


def test_precomputed_kernel_returns_K():
    K = np.array([[1, 2], [3, 4]], dtype=float)
    k = Precomputed(K)
    X = np.zeros((2, 2), dtype=float)
    assert np.allclose(k(X, X), K)


def test_kernel_combinators():
    X = np.array([[1, 0], [0, 1]], dtype=float)
    lin = Linear()
    rbf = RBF(gamma=1.0)

    scale = Scale(2.0, lin)
    sum_k = Sum(lin, rbf)
    prod_k = Product(lin, rbf)
    norm_k = Normalize(lin)

    K_lin = lin(X, X)
    K_rbf = rbf(X, X)

    assert np.allclose(scale(X, X), 2.0 * K_lin)
    assert np.allclose(sum_k(X, X), K_lin + K_rbf)
    assert np.allclose(prod_k(X, X), K_lin * K_rbf)

    cos_expected = K_lin / (np.sqrt(np.diag(K_lin))[:, None] * np.sqrt(np.diag(K_lin))[None, :] + 1e-12)
    assert np.allclose(norm_k(X, X), cos_expected)


def test_get_kernel_resolver_handles_variants_and_errors():
    X = np.eye(2)

    lin = Linear()
    assert get_kernel(lin) is lin

    k_fn = lambda A, B: A @ B.T
    k = get_kernel(k_fn)
    assert np.allclose(k(X, X), k_fn(X, X))

    k_name = get_kernel("linear")
    assert isinstance(k_name, Linear)

    K = np.array([[1, 2], [3, 4]], dtype=float)
    k_pre = get_kernel("precomputed", K=K)
    assert np.allclose(k_pre(X, X), K)

    with pytest.raises(ValueError):
        get_kernel("precomputed")
    with pytest.raises(ValueError):
        get_kernel("unknown")