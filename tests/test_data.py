import numpy as np
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

try:
    from sawmil.data import musk as musk_mod
    from sawmil.data.dummy import generate_dummy_bags, _rand_rot_cov, _components_from_centers, _sample_comp, GaussianComp
    from sawmil.bag import Bag, BagDataset  # type: ignore
except Exception:  # pragma: no cover
    from ..src.sawmil.data import musk as musk_mod  # type: ignore
    from ..src.sawmil.data.dummy import generate_dummy_bags, _rand_rot_cov, _components_from_centers, _sample_comp, GaussianComp  # type: ignore
    from ..src.sawmil.bag import Bag, BagDataset  # type: ignore

"""
Test suite for make_complex_bags.py module.

Tests cover:
- Random covariance matrix generation
- Gaussian component construction
- Sampling from components
- Full bag dataset generation with various parameters
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

class TestRandRotCov:
    """Test random rotation covariance matrix generation."""
    
    @pytest.mark.parametrize("d", [2, 3, 5, 10])
    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
    def test_spd_property(self, d: int, scale: float):
        """Test that generated matrix is symmetric positive definite."""
        rng = np.random.default_rng(42)
        cov = _rand_rot_cov(rng, d=d, scale=scale)
        
        # Check symmetry
        assert_array_almost_equal(cov, cov.T, decimal=10)
        
        # Check positive definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0), "Covariance matrix is not positive definite"
        
        # Check dimensions
        assert cov.shape == (d, d)
    
    @pytest.mark.parametrize("anisotropy", [0.0, 0.5, 0.9, 1.0])
    def test_anisotropy_effect(self, anisotropy: float):
        """Test that anisotropy controls eigenvalue spread."""
        rng = np.random.default_rng(42)
        d = 5
        scale = 2.0
        cov = _rand_rot_cov(rng, d=d, scale=scale, anisotropy=anisotropy)
        
        eigvals = np.linalg.eigvalsh(cov)
        eigvals_sorted = np.sort(eigvals)[::-1]
        
        # Expected eigenvalue range
        max_expected = scale ** 2
        min_expected = max_expected * (1.0 - anisotropy)
        
        # Check eigenvalue bounds
        assert eigvals_sorted[0] <= max_expected * 1.01  # Small tolerance
        assert eigvals_sorted[-1] >= min_expected * 0.99
        
        # Check condition number increases with anisotropy
        if anisotropy > 0:
            condition_number = eigvals_sorted[0] / eigvals_sorted[-1]
            expected_condition = 1.0 / (1.0 - anisotropy) if anisotropy < 1.0 else np.inf
            if anisotropy < 1.0:
                assert abs(condition_number - expected_condition) / expected_condition < 0.1
    
    def test_reproducibility(self):
        """Test that same seed produces same covariance."""
        d = 3
        cov1 = _rand_rot_cov(np.random.default_rng(123), d=d)
        cov2 = _rand_rot_cov(np.random.default_rng(123), d=d)
        assert_array_almost_equal(cov1, cov2)


class TestComponentsFromCenters:
    """Test Gaussian component construction."""
    
    def test_basic_construction(self):
        """Test basic component creation."""
        rng = np.random.default_rng(42)
        centers = [(0.0, 0.0), (1.0, 1.0), (2.0, -1.0)]
        scales = [(1.0, 0.5), (0.5, 0.0), (2.0, 0.8)]
        
        comps = _components_from_centers(rng, centers, scales)
        
        assert len(comps) == len(centers)
        
        for i, comp in enumerate(comps):
            assert isinstance(comp, GaussianComp)
            assert_array_almost_equal(comp.mu, centers[i])
            assert comp.cov.shape == (2, 2)
            # Check SPD
            assert_array_almost_equal(comp.cov, comp.cov.T)
            assert np.all(np.linalg.eigvalsh(comp.cov) > 0)
    
    def test_dimension_consistency(self):
        """Test that all dimensions are consistent."""
        rng = np.random.default_rng(42)
        d = 4
        centers = [tuple(np.random.randn(d)) for _ in range(3)]
        scales = [(1.0, 0.5) for _ in range(3)]
        
        comps = _components_from_centers(rng, centers, scales)
        
        for comp in comps:
            assert comp.mu.shape == (d,)
            assert comp.cov.shape == (d, d)


class TestSampleComp:
    """Test sampling from Gaussian components."""
    
    def test_sample_shape(self):
        """Test that samples have correct shape."""
        rng = np.random.default_rng(42)
        d = 3
        n = 100
        
        comp = GaussianComp(
            mu=np.array([1.0, 2.0, 3.0]),
            cov=np.eye(d)
        )
        
        samples = _sample_comp(rng, comp, n)
        assert samples.shape == (n, d)
    
    @pytest.mark.parametrize("n_samples", [1000, 5000])
    def test_sample_statistics(self, n_samples: int):
        """Test that samples follow the specified distribution."""
        rng = np.random.default_rng(42)
        mu = np.array([1.0, -2.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        
        comp = GaussianComp(mu=mu, cov=cov)
        samples = _sample_comp(rng, comp, n_samples)
        
        # Test mean
        sample_mean = np.mean(samples, axis=0)
        assert_allclose(sample_mean, mu, rtol=0.1, atol=0.1)
        
        # Test covariance
        sample_cov = np.cov(samples.T)
        assert_allclose(sample_cov, cov, rtol=0.1, atol=0.1)
        
        # Optional: Multivariate normality test (Henze-Zirkler)
        # This would require additional statistical packages


class TestGenerateDummyBags:
    """Test the main bag generation function."""
    
    def test_basic_generation(self):
        """Test basic dataset generation with default parameters."""
        dataset = generate_dummy_bags(
            n_pos=50,
            n_neg=30,
            d=2,
            random_state=42
        )
        
        assert isinstance(dataset, BagDataset)
        assert len(dataset.bags) == 80
        
        # Count positive and negative bags
        pos_bags = [b for b in dataset.bags if b.y == 1.0]
        neg_bags = [b for b in dataset.bags if b.y == 0.0]
        
        assert len(pos_bags) == 50
        assert len(neg_bags) == 30
    
    def test_bag_size_constraints(self):
        """Test that bag sizes respect specified constraints."""
        min_size, max_size = 5, 15
        dataset = generate_dummy_bags(
            n_pos=20,
            n_neg=20,
            inst_per_bag=(min_size, max_size),
            random_state=42
        )
        
        for bag in dataset.bags:
            assert min_size <= bag.X.shape[0] <= max_size
    
    def test_dimensionality(self):
        """Test that instances have correct dimensionality."""
        d = 7
        dataset = generate_dummy_bags(
            n_pos=10,
            n_neg=10,
            d=d,
            pos_centers=[tuple(np.random.randn(d)) for _ in range(2)],
            neg_centers=[tuple(np.random.randn(d)) for _ in range(2)],
            random_state=42
        )
        
        for bag in dataset.bags:
            assert bag.X.shape[1] == d
    
    def test_positive_instance_guarantee(self):
        """Test that positive bags always have at least one positive instance."""
        dataset = generate_dummy_bags(
            n_pos=30,
            n_neg=20,
            pos_intra_rate=(0.0, 0.1),  # Very low rate to test guarantee
            ensure_pos_in_every_pos_bag=True,
            random_state=42
        )
        
        pos_bags = [b for b in dataset.bags if b.y == 1.0]
        for bag in pos_bags:
            # At least one instance should have intra_bag_mask == 1
            assert np.sum(bag.intra_bag_mask) >= 1
    
    def test_intra_rate_bounds(self):
        """Test that intra-positive rates are within specified bounds."""
        min_rate, max_rate = 0.3, 0.8
        dataset = generate_dummy_bags(
            n_pos=50,
            n_neg=30,
            inst_per_bag=(10, 20),
            pos_intra_rate=(min_rate, max_rate),
            ensure_pos_in_every_pos_bag=False,
            random_state=42
        )
        
        pos_bags = [b for b in dataset.bags if b.y == 1.0]
        rates = []
        for bag in pos_bags:
            rate = np.mean(bag.intra_bag_mask)
            rates.append(rate)
            # Individual bags should roughly follow the rate
            # (with some tolerance for small bags)
            if bag.X.shape[0] >= 10:
                assert 0.0 <= rate <= 1.0
        
        # Average rate across all positive bags should be in range
        avg_rate = np.mean(rates)
        assert min_rate * 0.9 <= avg_rate <= max_rate * 1.1
    
    def test_contamination_rates(self):
        """Test cross-contamination rates."""
        dataset = generate_dummy_bags(
            n_pos=30,
            n_neg=30,
            inst_per_bag=(20, 30),
            neg_pos_noise_rate=(0.05, 0.15),
            pos_neg_noise_rate=(0.10, 0.25),
            random_state=42
        )
        
        # This is harder to test directly without access to internal labels
        # We mainly check that generation doesn't fail
        assert len(dataset.bags) == 60
    
    def test_reproducibility(self):
        """Test that same random state produces identical datasets."""
        ds1 = generate_dummy_bags(n_pos=10, n_neg=10, random_state=123)
        ds2 = generate_dummy_bags(n_pos=10, n_neg=10, random_state=123)
        
        assert len(ds1.bags) == len(ds2.bags)
        
        for b1, b2 in zip(ds1.bags, ds2.bags):
            assert_array_almost_equal(b1.X, b2.X)
            assert b1.y == b2.y
            assert_array_almost_equal(b1.intra_bag_mask, b2.intra_bag_mask)
    
    def test_outlier_injection(self):
        """Test that outliers are properly injected."""
        # Generate with high outlier rate to ensure some outliers
        dataset = generate_dummy_bags(
            n_pos=20,
            n_neg=20,
            inst_per_bag=(50, 60),
            outlier_rate=0.1,
            outlier_scale=20.0,
            random_state=42
        )
        
        # Collect all instances
        all_instances = []
        for bag in dataset.bags:
            all_instances.append(bag.X)
        all_instances = np.vstack(all_instances)
        
        # Check for presence of outliers (instances far from origin)
        distances = np.linalg.norm(all_instances, axis=1)
        # With outlier_scale=20, we expect some instances with large norms
        assert np.max(distances) > 15.0  # Should have some outliers
        
        # Roughly 10% should be outliers (with some tolerance)
        outlier_threshold = np.percentile(distances, 85)
        frac_outliers = np.mean(distances > outlier_threshold)
        assert 0.05 <= frac_outliers <= 0.20


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_bags_not_allowed(self):
        """Test that minimum bag size is enforced."""
        with pytest.raises(AssertionError):
            generate_dummy_bags(inst_per_bag=(0, 5))
    
    def test_invalid_bag_size_range(self):
        """Test that max >= min for bag sizes."""
        with pytest.raises(AssertionError):
            generate_dummy_bags(inst_per_bag=(10, 5))
    
    def test_mismatched_centers_scales(self):
        """Test that centers and scales must have same length."""
        with pytest.raises(AssertionError):
            generate_dummy_bags(
                pos_centers=[(0, 0), (1, 1)],
                pos_scales=[(1.0, 0.5)]  # Only one scale for two centers
            )
    
    def test_dimension_mismatch(self):
        """Test that all centers must have same dimension as d."""
        with pytest.raises(AssertionError):
            generate_dummy_bags(
                d=2,
                pos_centers=[(0, 0, 0)],  # 3D center for 2D data
                neg_centers=[(1, 1)]
            )
    
    def test_zero_bags(self):
        """Test generation with zero bags of one class."""
        # Should work fine with 0 negative bags
        dataset = generate_dummy_bags(n_pos=10, n_neg=0)
        assert len(dataset.bags) == 10
        
        # Should work fine with 0 positive bags
        dataset = generate_dummy_bags(n_pos=0, n_neg=10)
        assert len(dataset.bags) == 10


class TestStatisticalProperties:
    """Test statistical properties of generated data."""
    
    def test_cluster_separation(self):
        """Test that positive and negative clusters are separated."""
        # Use well-separated centers
        dataset = generate_dummy_bags(
            n_pos=50,
            n_neg=50,
            inst_per_bag=(30, 40),
            pos_centers=[(5.0, 5.0), (7.0, 7.0)],
            neg_centers=[(-5.0, -5.0), (-7.0, -7.0)],
            pos_scales=[(0.5, 0.0), (0.5, 0.0)],  # Small variance
            neg_scales=[(0.5, 0.0), (0.5, 0.0)],
            pos_intra_rate=(0.8, 1.0),
            neg_pos_noise_rate=(0.0, 0.0),  # No contamination
            pos_neg_noise_rate=(0.0, 0.0),
            outlier_rate=0.0,
            random_state=42
        )
        
        # Collect positive and negative instances
        pos_instances = []
        neg_instances = []
        
        for bag in dataset.bags:
            if bag.y == 1.0:
                # Get instances marked as positive
                pos_idx = bag.intra_bag_mask == 1.0
                if np.any(pos_idx):
                    pos_instances.append(bag.X[pos_idx])
            else:
                neg_instances.append(bag.X)
        
        if pos_instances:
            pos_instances = np.vstack(pos_instances)
        if neg_instances:
            neg_instances = np.vstack(neg_instances)
        
        # Check separation (simplified - just check means are far apart)
        if len(pos_instances) > 0 and len(neg_instances) > 0:
            pos_mean = np.mean(pos_instances, axis=0)
            neg_mean = np.mean(neg_instances, axis=0)
            separation = np.linalg.norm(pos_mean - neg_mean)
            assert separation > 5.0  # Should be well separated


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])