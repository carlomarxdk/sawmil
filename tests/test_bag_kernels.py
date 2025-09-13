import numpy as np
import pytest

try:
    from sawmil.bag_kernels import (
        WeightedMeanBagKernel,
        PrecomputedBagKernel,
        make_bag_kernel,
    )
    from sawmil.kernels import Linear, RBF
    from sawmil.bag import Bag
except Exception:  # pragma: no cover
    from ..src.sawmil.bag_kernels import (
        WeightedMeanBagKernel,
        PrecomputedBagKernel,
        make_bag_kernel,
    )
    from ..src.sawmil.kernels import Linear, RBF
    from ..src.sawmil.bag import Bag


def _bag(X, y):
    # minimal helper: Bag(X) should expose .X, .n, .d
    return Bag(X, y)

def test_weighted_mean_bag_kernel_normalizers():
    bag1 = Bag(X=np.array([[1.0], [2.0]]), y=1.0)
    bag2 = Bag(X=np.array([[3.0], [4.0]]), y=-1.0)

    k_none = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="none")
    k_avg = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="average")
    k_feat = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="featurespace")

    K_none = k_none([bag1], [bag2])
    K_avg = k_avg([bag1], [bag2])
    K_feat = k_feat([bag1], [bag2])

    assert np.allclose(K_none, np.array([[21.0]]))
    assert np.allclose(K_avg, np.array([[1.3125]]))
    assert np.allclose(K_feat, np.array([[1.0]]))


def test_weighted_mean_bag_kernel_exponent_clamps_negatives():
    bag_pos = Bag(X=np.array([[1.0]]), y=1.0)
    bag_neg = Bag(X=np.array([[-1.0]]), y=-1.0)
    k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="none", p=2.0)
    K = k([bag_pos], [bag_neg])
    assert np.allclose(K, np.array([[0.0]]))


def test_weighted_mean_bag_kernel_fit_sets_rbf_gamma_and_computes():
    bag1 = Bag(X=np.array([[1.0, 0.0], [0.0, 1.0]]), y=1.0)
    bag2 = Bag(X=np.array([[1.0, 1.0]]), y=-1.0)
    k = WeightedMeanBagKernel(inst_kernel=RBF(), normalizer="none")
    k.fit([bag1, bag2])
    assert k.inst_kernel.gamma == pytest.approx(0.5)
    K = k([bag1], [bag2])
    expected = 2.0 * np.exp(-0.5)
    assert np.allclose(K, np.array([[expected]]))


def test_precomputed_bag_kernel_returns_matrix():
    K = np.array([[1.0, 2.0], [3.0, 4.0]])
    bk = PrecomputedBagKernel(K)
    bag_a = Bag(X=np.array([[0.0]]), y=1.0)
    bag_b = Bag(X=np.array([[0.0]]), y=-1.0)
    bags = [bag_a, bag_b]
    assert np.allclose(bk(bags, bags), K)


def test_make_bag_kernel_factory_returns_configured_instance():
    bk = make_bag_kernel(Linear(), normalizer="featurespace", p=2.0)
    assert isinstance(bk, WeightedMeanBagKernel)
    assert bk.normalizer == "featurespace"
    assert bk.p == 2.0


def test_bag_kernel_symmetry_and_nonneg_after_power():
    rng = np.random.default_rng(0)
    bags = [ _bag(rng.normal(size=(m, 4)), y=1.0) for m in (1, 2, 3) ]
    k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="none", p=2.0)
    K = k(bags, bags)
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.all(K >= -1e-14) 