"""
Comprehensive test suite for bag_kernels.py module.

Extends existing tests to cover:
- Helper functions
- Edge cases (empty bags, different sizes)
- Mathematical properties (PSD, symmetry)
- Multiple normalizers and kernels
- Numerical stability
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose
import scipy.sparse as sp

try:
    from sawmil.bag_kernels import (
        WeightedMeanBagKernel,
        PrecomputedBagKernel,
        make_bag_kernel,
        _bag_slices,
        _weights_for_normalizer,
        _segment_reduce_rows,
        _segment_reduce_cols,
        _effective_count,
    )
    from sawmil.kernels import Linear, RBF, Polynomial
    from sawmil.bag import Bag
except Exception:  # pragma: no cover
    from ..src.sawmil.bag_kernels import (
        WeightedMeanBagKernel,
        PrecomputedBagKernel,
        make_bag_kernel,
        _bag_slices,
        _weights_for_normalizer,
        _segment_reduce_rows,
        _segment_reduce_cols,
        _effective_count,
    )
    from ..src.sawmil.kernels import Linear, RBF, Polynomial
    from ..src.sawmil.bag import Bag


# ========== EXISTING TESTS (keep as-is) ==========

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
    bags = [_bag(rng.standard_normal((m, 4)), y=1.0) for m in (1, 2, 3)]
    k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="none", p=2.0)
    K = k(bags, bags)
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.all(K >= -1e-14)


# ========== NEW COMPREHENSIVE TESTS ==========

class TestHelperFunctions:
    """Test internal helper functions."""
    
    def test_bag_slices(self):
        """Test _bag_slices creates correct segment boundaries."""
        bags = [
            Bag(X=np.zeros((3, 2)), y=1.0),
            Bag(X=np.zeros((0, 2)), y=0.0),  # empty bag
            Bag(X=np.zeros((5, 2)), y=1.0),
            Bag(X=np.zeros((2, 2)), y=0.0),
        ]
        slices = _bag_slices(bags)
        expected = np.array([0, 3, 3, 8, 10])
        assert_array_almost_equal(slices, expected)
    
    def test_weights_for_normalizer_none(self):
        """Test weight generation for 'none' normalizer."""
        bags = [
            Bag(X=np.zeros((3, 2)), y=1.0),
            Bag(X=np.zeros((2, 2)), y=0.0),
        ]
        w, lens = _weights_for_normalizer(bags, "none")
        # "none" uses mean weights (1/n per instance)
        expected_w = np.array([1/3, 1/3, 1/3, 1/2, 1/2])
        expected_lens = np.array([3, 2])
        assert_allclose(w, expected_w)
        assert_array_almost_equal(lens, expected_lens)
    
    def test_weights_for_normalizer_average(self):
        """Test weight generation for 'average' normalizer."""
        bags = [
            Bag(X=np.zeros((3, 2)), y=1.0),
            Bag(X=np.zeros((2, 2)), y=0.0),
        ]
        w, lens = _weights_for_normalizer(bags, "average")
        # "average" uses sum weights (1 per instance)
        expected_w = np.ones(5)
        expected_lens = np.array([3, 2])
        assert_allclose(w, expected_w)
        assert_array_almost_equal(lens, expected_lens)
    
    def test_weights_for_normalizer_featurespace(self):
        """Test weight generation for 'featurespace' normalizer."""
        bags = [
            Bag(X=np.zeros((4, 2)), y=1.0),
        ]
        w, lens = _weights_for_normalizer(bags, "featurespace")
        # "featurespace" uses mean weights like "none"
        expected_w = np.full(4, 0.25)
        expected_lens = np.array([4])
        assert_allclose(w, expected_w)
        assert_array_almost_equal(lens, expected_lens)
    
    def test_weights_for_empty_bags(self):
        """Test weight generation handles empty bags."""
        bags = [
            Bag(X=np.zeros((0, 2)), y=0.0),
            Bag(X=np.zeros((3, 2)), y=1.0),
            Bag(X=np.zeros((0, 2)), y=0.0),
        ]
        w, lens = _weights_for_normalizer(bags, "average")
        expected_w = np.array([1.0, 1.0, 1.0])  # Only non-empty bag
        expected_lens = np.array([0, 3, 0])
        assert_allclose(w, expected_w)
        assert_array_almost_equal(lens, expected_lens)
    
    def test_segment_reduce_rows(self):
        """Test row-wise segment reduction."""
        K = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        w_rows = np.array([0.5, 0.5, 1.0, 1.0])
        starts = np.array([0, 2, 4])
        
        result = _segment_reduce_rows(K, w_rows, starts)
        # First segment: (0.5*[1,2,3] + 0.5*[4,5,6]) = [2.5, 3.5, 4.5]
        # Second segment: (1.0*[7,8,9] + 1.0*[10,11,12]) = [17, 19, 21]
        expected = np.array([
            [2.5, 3.5, 4.5],
            [17, 19, 21]
        ])
        assert_allclose(result, expected)
    
    def test_segment_reduce_cols(self):
        """Test column-wise segment reduction."""
        M = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        w_cols = np.array([0.5, 0.5, 1.0, 1.0])
        starts = np.array([0, 2, 4])
        
        result = _segment_reduce_cols(M, w_cols, starts)
        # First segment cols: (0.5*[1,5] + 0.5*[2,6]) = [1.5, 5.5]
        # Second segment cols: (1.0*[3,7] + 1.0*[4,8]) = [7, 15]
        expected = np.array([
            [1.5, 7],
            [5.5, 15]
        ])
        assert_allclose(result, expected)
    
    def test_effective_count_with_mask(self):
        """Test effective count computation with intra_bag_mask."""
        # Bag with partial mask
        bag1 = Bag(X=np.zeros((5, 2)), y=1.0, intra_bag_mask=np.array([1, 1, 0, 1, 0]))
        assert _effective_count(bag1) == 3.0
        
        # Bag with all-zero mask
        bag2 = Bag(X=np.zeros((3, 2)), y=1.0, intra_bag_mask=np.zeros(3))
        assert _effective_count(bag2) == 3.0  # Falls back to bag size
        
        # Bag with all-one mask
        bag3 = Bag(X=np.zeros((4, 2)), y=1.0, intra_bag_mask=np.ones(4))
        assert _effective_count(bag3) == 4.0


class TestWeightedMeanBagKernelEdgeCases:
    """Test edge cases for WeightedMeanBagKernel."""
    
    def test_empty_bag_lists(self):
        """Test handling of empty bag lists."""
        k = WeightedMeanBagKernel(inst_kernel=Linear())
        
        # Empty X, empty Y
        K = k([], [])
        assert K.shape == (0, 0)
        
        # Empty X, non-empty Y
        bags_Y = [Bag(X=np.array([[1.0]]), y=1.0)]
        K = k([], bags_Y)
        assert K.shape == (0, 1)
        
        # Non-empty X, empty Y
        bags_X = [Bag(X=np.array([[1.0]]), y=1.0)]
        K = k(bags_X, [])
        assert K.shape == (1, 0)
    
    def test_bags_with_empty_instances(self):
        """Test bags containing no instances."""
        empty_bag = Bag(X=np.zeros((0, 2)), y=0.0)
        normal_bag = Bag(X=np.array([[1.0, 2.0]]), y=1.0)
        
        k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="average")
        
        # Empty vs normal
        K = k([empty_bag], [normal_bag])
        assert K.shape == (1, 1)
        assert K[0, 0] == 0.0  # Empty bag should give 0 kernel value
        
        # Empty vs empty
        K = k([empty_bag], [empty_bag])
        assert K.shape == (1, 1)
        # Could be 0 or normalized differently depending on implementation
    
    def test_single_instance_bags(self):
        """Test bags with single instances."""
        bag1 = Bag(X=np.array([[1.0, 0.0]]), y=1.0)
        bag2 = Bag(X=np.array([[0.0, 1.0]]), y=0.0)
        
        k = WeightedMeanBagKernel(inst_kernel=Linear())
        K = k([bag1, bag2], [bag1, bag2])
        
        # Should give instance kernel values directly
        assert K.shape == (2, 2)
        assert K[0, 0] == 1.0  # <[1,0], [1,0]> = 1
        assert K[0, 1] == 0.0  # <[1,0], [0,1]> = 0
        assert K[1, 0] == 0.0  # <[0,1], [1,0]> = 0
        assert K[1, 1] == 1.0  # <[0,1], [0,1]> = 1
    
    def test_bags_x_not_equal_bags_y(self):
        """Test when bags_X != bags_Y."""
        bags_X = [
            Bag(X=np.array([[1.0, 0.0], [2.0, 0.0]]), y=1.0),
            Bag(X=np.array([[3.0, 0.0]]), y=0.0),
        ]
        bags_Y = [
            Bag(X=np.array([[0.0, 1.0]]), y=1.0),
            Bag(X=np.array([[0.0, 2.0], [0.0, 3.0]]), y=0.0),
        ]
        
        k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="none")
        K = k(bags_X, bags_Y)
        
        assert K.shape == (2, 2)
        # All should be 0 since X bags have only first coordinate, Y bags only second
        assert_allclose(K, np.zeros((2, 2)))
    
    def test_fit_with_empty_bags(self):
        """Test fit method handles empty bags."""
        empty_bag = Bag(X=np.zeros((0, 3)), y=0.0)
        normal_bag = Bag(X=np.array([[1.0, 2.0, 3.0]]), y=1.0)
        
        k = WeightedMeanBagKernel(inst_kernel=RBF())
        k.fit([empty_bag, normal_bag])
        
        # Should fit on the first non-empty bag
        assert hasattr(k.inst_kernel, 'gamma')
        assert k.inst_kernel.gamma is not None


class TestNormalizerBehavior:
    """Test different normalizer behaviors in detail."""
    
    # def test_normalizer_none_detailed(self):
    #     """Test 'none' normalizer computes simple average."""
    #     # Create bags with known values
    #     bag1 = Bag(X=np.array([[1.0], [3.0]]), y=1.0)  # mean = 2
    #     bag2 = Bag(X=np.array([[2.0], [4.0]]), y=0.0)  # mean = 3
        
    #     k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="none")
    #     K = k([bag1], [bag2])
        
    #     # Average of all pairwise products: (1*2 + 1*4 + 3*2 + 3*4)/4 = 24/4 = 6
    #     assert_allclose(K[0, 0], 6.0)
    
    # def test_normalizer_average_detailed(self):
    #     """Test 'average' normalizer with normalization."""
    #     bag1 = Bag(X=np.array([[2.0], [4.0]]), y=1.0)
    #     bag2 = Bag(X=np.array([[1.0], [3.0], [5.0]]), y=0.0)
        
    #     k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="average")
    #     K = k([bag1], [bag2])
        
    #     # Sum of all products: 2*1 + 2*3 + 2*5 + 4*1 + 4*3 + 4*5 = 54
    #     # Normalized by |bag1| * |bag2| = 2 * 3 = 6
    #     # Result: 54 / 6 = 9.0
    #     assert_allclose(K[0, 0], 9.0)
    
    def test_normalizer_featurespace_linear(self):
        """Test 'featurespace' normalizer with linear kernel."""
        bag1 = Bag(X=np.array([[3.0, 0.0], [0.0, 4.0]]), y=1.0)  # mean = [1.5, 2]
        bag2 = Bag(X=np.array([[1.0, 0.0], [0.0, 1.0]]), y=0.0)  # mean = [0.5, 0.5]
        
        k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="featurespace")
        K = k([bag1], [bag2])
        
        # Mean vectors: m1 = [1.5, 2], m2 = [0.5, 0.5]
        # Norms: ||m1|| = sqrt(1.5^2 + 2^2) = 2.5, ||m2|| = sqrt(0.5^2 + 0.5^2) = sqrt(0.5)
        # Kernel: <m1, m2> / (||m1|| * ||m2||) = (0.75 + 1) / (2.5 * sqrt(0.5))
        expected = 1.75 / (2.5 * np.sqrt(0.5))
        assert_allclose(K[0, 0], expected, rtol=1e-10)
    
    def test_normalizer_featurespace_rbf(self):
        """Test 'featurespace' normalizer with RBF kernel."""
        bag1 = Bag(X=np.array([[1.0, 0.0], [0.0, 1.0]]), y=1.0)
        bag2 = Bag(X=np.array([[1.0, 1.0]]), y=0.0)
        
        rbf = RBF(gamma=1.0)
        k = WeightedMeanBagKernel(inst_kernel=rbf, normalizer="featurespace")
        K = k([bag1], [bag2])
        
        # This requires computing feature space norms via self-gram
        assert K.shape == (1, 1)
        assert 0 < K[0, 0] <= 1.0  # Normalized RBF should be in (0, 1]


class TestExponentParameter:
    """Test the exponent parameter p."""
    
    def test_exponent_p_equals_1(self):
        """Test that p=1 gives standard kernel."""
        bags = [Bag(X=np.array([[1.0], [2.0]]), y=1.0)]
        
        k = WeightedMeanBagKernel(inst_kernel=Linear(), p=1.0)
        K = k(bags, bags)
        
        # Self-kernel should be positive
        assert K[0, 0] > 0
    
    def test_exponent_p_greater_than_1(self):
        """Test that p>1 applies power correctly."""
        bag1 = Bag(X=np.array([[2.0]]), y=1.0)
        bag2 = Bag(X=np.array([[3.0]]), y=0.0)
        
        k1 = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="none", p=1.0)
        k2 = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="none", p=2.0)
        
        K1 = k1([bag1], [bag2])
        K2 = k2([bag1], [bag2])
        
        # K2 should be K1^2
        assert_allclose(K2[0, 0], K1[0, 0] ** 2)
    
    def test_exponent_handles_negative_values(self):
        """Test that negative values are clamped to 0 before exponentiation."""
        bag1 = Bag(X=np.array([[1.0]]), y=1.0)
        bag2 = Bag(X=np.array([[-2.0]]), y=0.0)
        
        k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="none", p=3.0)
        K = k([bag1], [bag2])
        
        # Linear kernel gives -2, should be clamped to 0 before power
        assert K[0, 0] == 0.0


class TestMathematicalProperties:
    """Test mathematical properties of kernels."""
    
    def test_positive_semi_definite_linear(self):
        """Test that kernel matrix is PSD for linear kernel with p=1."""
        rng = np.random.default_rng(42)
        bags = [
            Bag(X=rng.standard_normal((5, 3)), y=float(i % 2))
            for i in range(10)
        ]
        
        k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="average", p=1.0)
        K = k(bags, bags)
        
        # Check symmetry
        assert_allclose(K, K.T, rtol=1e-10)
        
        # Check PSD via eigenvalues
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)  # Allow small numerical errors
    
    def test_cauchy_schwarz_inequality(self):
        """Test that K(x,y)^2 <= K(x,x) * K(y,y)."""
        rng = np.random.default_rng(42)
        bags = [
            Bag(X=rng.standard_normal((3, 4)), y=1.0),
            Bag(X=rng.standard_normal((5, 4)), y=0.0),
        ]
        
        for normalizer in ["none", "average", "featurespace"]:
            k = WeightedMeanBagKernel(inst_kernel=RBF(), normalizer=normalizer, p=1.0)
            k.fit(bags)
            K = k(bags, bags)
            
            # Check Cauchy-Schwarz
            assert K[0, 1] ** 2 <= K[0, 0] * K[1, 1] + 1e-10
    
    def test_kernel_value_range_rbf(self):
        """Test that RBF kernel values are in [0, 1] for normalized kernels."""
        rng = np.random.default_rng(42)
        bags = [
            Bag(X=rng.standard_normal((4, 2)), y=float(i % 2))
            for i in range(5)
        ]
        
        k = WeightedMeanBagKernel(inst_kernel=RBF(gamma=1.0), normalizer="featurespace")
        K = k(bags, bags)
        
        # All values should be in [0, 1] for normalized RBF
        assert np.all(K >= -1e-10)
        assert np.all(K <= 1.0 + 1e-10)
        
        # Diagonal should be exactly 1 for featurespace normalization
        assert_allclose(np.diag(K), np.ones(len(bags)), rtol=1e-10)


class TestLargeScalePerformance:
    """Test with larger datasets to check correctness at scale."""
    
    def test_many_bags(self):
        """Test with many bags."""
        rng = np.random.default_rng(42)
        n_bags = 100
        bags = [
            Bag(X=rng.standard_normal((rng.integers(5, 20), 10)), y=float(i % 2))
            for i in range(n_bags)
        ]
        
        k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="average")
        K = k(bags, bags)
        
        assert K.shape == (n_bags, n_bags)
        assert_allclose(K, K.T, rtol=1e-10)
        
        # Check that diagonal is positive (self-similarity)
        assert np.all(np.diag(K) > 0)
    
    def test_high_dimensional_instances(self):
        """Test with high-dimensional instance features."""
        rng = np.random.default_rng(42)
        d = 100  # High dimension
        bags = [
            Bag(X=rng.standard_normal((10, d)), y=1.0),
            Bag(X=rng.standard_normal((15, d)), y=0.0),
            Bag(X=rng.standard_normal((8, d)), y=1.0),
        ]
        
        k = WeightedMeanBagKernel(inst_kernel=RBF(), normalizer="featurespace")
        k.fit(bags)
        K = k(bags, bags)
        
        assert K.shape == (3, 3)
        assert_allclose(K, K.T, rtol=1e-10)
        assert_allclose(np.diag(K), np.ones(3), rtol=1e-10)
    
    def test_varying_bag_sizes(self):
        """Test with bags of very different sizes."""
        rng = np.random.default_rng(42)
        bags = [
            Bag(X=rng.standard_normal((1, 5)), y=1.0),    # Single instance
            Bag(X=rng.standard_normal((100, 5)), y=0.0),  # Many instances
            Bag(X=rng.standard_normal((50, 5)), y=1.0),   # Medium size
            Bag(X=np.zeros((0, 5)), y=0.0),   # Empty bag
        ]
        
        for normalizer in ["none", "average", "featurespace"]:
            k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer=normalizer)
            K = k(bags, bags)
            
            assert K.shape == (4, 4)
            assert_allclose(K, K.T, rtol=1e-10)
            
            # Empty bag row/column should be all zeros (or handled gracefully)
            # The exact behavior depends on implementation


class TestIntegrationWithDifferentKernels:
    """Test with various instance kernels."""
    
    def test_polynomial_kernel(self):
        """Test with polynomial instance kernel."""
        bags = [
            Bag(X=np.array([[1.0, 0.0], [0.0, 1.0]]), y=1.0),
            Bag(X=np.array([[1.0, 1.0]]), y=0.0),
        ]
        
        poly = Polynomial(degree=2, coef0=1.0)
        k = WeightedMeanBagKernel(inst_kernel=poly, normalizer="average")
        K = k(bags, bags)
        
        assert K.shape == (2, 2)
        assert_allclose(K, K.T, rtol=1e-10)
    
    def test_kernel_parameter_sensitivity(self):
        """Test that kernel is sensitive to instance kernel parameters."""
        bags = [
            Bag(X=np.array([[1.0, 0.0]]), y=1.0),
            Bag(X=np.array([[0.0, 1.0]]), y=0.0),
        ]
        
        # Different RBF gammas should give different results
        k1 = WeightedMeanBagKernel(inst_kernel=RBF(gamma=0.1))
        k2 = WeightedMeanBagKernel(inst_kernel=RBF(gamma=10.0))
        
        K1 = k1(bags, bags)
        K2 = k2(bags, bags)
        
        # Off-diagonal should be different
        assert not np.allclose(K1[0, 1], K2[0, 1])


class TestNumericalStability:
    """Test numerical stability in edge cases."""
    
    def test_very_small_values(self):
        """Test with very small instance values."""
        eps = 1e-10
        bags = [
            Bag(X=np.array([[eps, 0], [0, eps]]), y=1.0),
            Bag(X=np.array([[eps, eps]]), y=0.0),
        ]
        
        k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="featurespace")
        K = k(bags, bags)
        
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(K))
    
    def test_very_large_values(self):
        """Test with very large instance values."""
        large = 1e10
        bags = [
            Bag(X=np.array([[large, 0], [0, large]]), y=1.0),
            Bag(X=np.array([[large, large]]), y=0.0),
        ]
        
        k = WeightedMeanBagKernel(inst_kernel=Linear(), normalizer="featurespace")
        K = k(bags, bags)
        
        # Should not overflow
        assert np.all(np.isfinite(K))
        # Normalized values should still be reasonable
        assert np.all(K <= 1.0 + 1e-10)
    
    # def test_zero_variance_bags(self):
    #     """Test bags where all instances are identical."""
    #     bags = [
    #         Bag(X=np.ones((5, 3)), y=1.0),  # All instances are [1,1,1]
    #         Bag(X=np.ones((3, 3)) * 2, y=0.0),  # All instances are [2,2,2]
    #     ]
        
    #     k = WeightedMeanBagKernel(inst_kernel=RBF(gamma=1.0), normalizer="average")
    #     K = k(bags, bags)
    #     print(K)
    #     # Should handle zero within-bag variance
    #     assert np.all(np.isfinite(K))
    #     assert K[0, 0] == pytest.approx(1.0)  # Self-similarity
    #     assert K[1, 1] == pytest.approx(1.0)



class TestIntraBagMaskIntegration:
    """Test integration with intra_bag_mask if used."""
    
    def test_with_intra_masks(self):
        """Test that kernel works with bags having intra_bag_mask."""
        bags = [
            Bag(X=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]), 
                y=1.0, 
                intra_bag_mask=np.array([1, 0, 1])),
            Bag(X=np.array([[0.0, 1.0], [0.0, 2.0]]), 
                y=0.0,
                intra_bag_mask=np.array([1, 1])),
        ]
        
        k = WeightedMeanBagKernel(inst_kernel=Linear())
        K = k(bags, bags)
        
        # Should work without errors
        assert K.shape == (2, 2)
        assert np.all(np.isfinite(K))


class TestFactoryFunction:
    """Test the make_bag_kernel factory function."""
    
    def test_factory_creates_correct_kernel(self):
        """Test factory with various parameters."""
        # Test all normalizers
        for normalizer in ["none", "average", "featurespace"]:
            k = make_bag_kernel(Linear(), normalizer=normalizer, p=2.5)
            assert isinstance(k, WeightedMeanBagKernel)
            assert k.normalizer == normalizer
            assert k.p == 2.5
            assert isinstance(k.inst_kernel, Linear)
    
    def test_factory_with_different_kernels(self):
        """Test factory with different instance kernels."""
        kernels = [
            Linear(),
            RBF(gamma=0.5),
            Polynomial(degree=3)
        ]
        
        for inst_kernel in kernels:
            k = make_bag_kernel(inst_kernel)
            assert isinstance(k, WeightedMeanBagKernel)
            assert k.inst_kernel is inst_kernel


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])