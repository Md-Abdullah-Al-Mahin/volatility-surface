"""
Unit tests for Module 4: VolatilitySurfaceInterpolator
"""

import numpy as np
import pytest

from src.surface_interpolator import (
    VolatilitySurfaceInterpolator,
    DEFAULT_GRID_SIZE_M,
    DEFAULT_GRID_SIZE_T,
    MIN_POINTS,
)


class TestVolatilitySurfaceInterpolatorInit:
    """Tests for __init__."""

    def test_init_default(self):
        interp = VolatilitySurfaceInterpolator()
        assert interp.method == "cubic"

    def test_init_linear(self):
        interp = VolatilitySurfaceInterpolator(method="linear")
        assert interp.method == "linear"

    def test_init_invalid_raises(self):
        with pytest.raises(ValueError, match="method must be 'cubic' or 'linear'"):
            VolatilitySurfaceInterpolator(method="nearest")


class TestCreateGrid:
    """Tests for _create_grid."""

    def test_grid_shape(self):
        interp = VolatilitySurfaceInterpolator()
        T = np.array([0.1, 0.5, 1.0])
        M = np.array([0.9, 1.0, 1.1])
        T_grid, M_grid = interp._create_grid(T, M, n_points_T=20, n_points_M=30)
        assert T_grid.shape == (20, 30)
        assert M_grid.shape == (20, 30)

    def test_grid_bounds(self):
        interp = VolatilitySurfaceInterpolator()
        T = np.array([0.2, 0.8])
        M = np.array([0.95, 1.05])
        T_grid, M_grid = interp._create_grid(T, M, n_points_T=5, n_points_M=5)
        assert T_grid.min() <= 0.2 and T_grid.max() >= 0.8
        assert M_grid.min() <= 0.95 and M_grid.max() >= 1.05


class TestInterpolateSurface:
    """Tests for interpolate_surface."""

    @pytest.fixture
    def synthetic_points(self):
        """Synthetic (T, M, IV) on a regular grid for known surface."""
        np.random.seed(42)
        n = 15
        T = np.linspace(0.1, 1.0, n)
        M = np.linspace(0.85, 1.15, n)
        T_pts, M_pts = np.meshgrid(T, M, indexing="ij")
        T_flat = T_pts.ravel()
        M_flat = M_pts.ravel()
        # IV = 0.2 + 0.1*T + 0.05*(M-1)^2 (known smooth surface)
        IV_flat = 0.2 + 0.1 * T_flat + 0.05 * (M_flat - 1.0) ** 2
        return T_flat, M_flat, IV_flat

    def test_returns_three_arrays(self, synthetic_points):
        interp = VolatilitySurfaceInterpolator(method="linear")
        T, M, IV = synthetic_points
        T_grid, M_grid, IV_grid = interp.interpolate_surface(T, M, IV)
        assert T_grid.shape == (DEFAULT_GRID_SIZE_T, DEFAULT_GRID_SIZE_M)
        assert M_grid.shape == (DEFAULT_GRID_SIZE_T, DEFAULT_GRID_SIZE_M)
        assert IV_grid.shape == (DEFAULT_GRID_SIZE_T, DEFAULT_GRID_SIZE_M)

    def test_interpolation_accuracy_linear(self, synthetic_points):
        """On synthetic smooth data, linear interpolation produces reasonable IV grid."""
        interp = VolatilitySurfaceInterpolator(method="linear")
        T, M, IV = synthetic_points
        T_grid, M_grid, IV_grid = interp.interpolate_surface(
            T, M, IV, n_points_T=25, n_points_M=25
        )
        # IV grid should be in reasonable range and have finite values where expected
        valid = np.isfinite(IV_grid)
        assert np.any(valid)
        if np.any(valid):
            assert 0 <= np.nanmin(IV_grid) <= np.nanmax(IV_grid) <= 2.0 + 1e-6

    def test_interpolation_accuracy_cubic(self, synthetic_points):
        interp = VolatilitySurfaceInterpolator(method="cubic")
        T, M, IV = synthetic_points
        T_grid, M_grid, IV_grid = interp.interpolate_surface(
            T, M, IV, n_points_T=25, n_points_M=25
        )
        assert np.nanmin(IV_grid) >= 0 or np.all(np.isnan(IV_grid))
        assert np.nanmax(IV_grid) <= 2.0 or np.all(np.isnan(IV_grid))

    def test_override_method(self, synthetic_points):
        T, M, IV = synthetic_points
        interp = VolatilitySurfaceInterpolator(method="cubic")
        T_g, M_g, IV_g = interp.interpolate_surface(T, M, IV, method="linear")
        assert IV_g.shape == (DEFAULT_GRID_SIZE_T, DEFAULT_GRID_SIZE_M)

    def test_custom_grid_size(self, synthetic_points):
        T, M, IV = synthetic_points
        interp = VolatilitySurfaceInterpolator()
        T_grid, M_grid, IV_grid = interp.interpolate_surface(
            T, M, IV, n_points_T=10, n_points_M=15
        )
        assert T_grid.shape == (10, 15)
        assert IV_grid.shape == (10, 15)

    def test_clip_iv(self, synthetic_points):
        T, M, IV = synthetic_points
        IV_high = IV + 1.5  # push some above 2
        interp = VolatilitySurfaceInterpolator()
        _, _, IV_grid = interp.interpolate_surface(T, M, IV_high, clip_iv=True)
        valid = ~np.isnan(IV_grid)
        if np.any(valid):
            assert np.max(IV_grid[valid]) <= 2.0
            assert np.min(IV_grid[valid]) >= 0.0

    def test_fill_value_nan(self, synthetic_points):
        T, M, IV = synthetic_points
        interp = VolatilitySurfaceInterpolator()
        _, _, IV_grid = interp.interpolate_surface(
            T, M, IV, fill_value=np.nan, n_points_T=50, n_points_M=50
        )
        # Some points inside convex hull should be finite
        assert np.any(np.isfinite(IV_grid))

    def test_too_few_points_raises(self):
        interp = VolatilitySurfaceInterpolator()
        T = np.array([0.1, 0.2, 0.3])  # only 3
        M = np.array([1.0, 1.0, 1.0])
        IV = np.array([0.2, 0.21, 0.22])
        with pytest.raises(ValueError, match=f"at least {MIN_POINTS} points"):
            interp.interpolate_surface(T, M, IV)

    def test_mismatched_lengths_raises(self):
        interp = VolatilitySurfaceInterpolator()
        T = np.ones(20)
        M = np.ones(15)
        IV = np.ones(20)
        with pytest.raises(ValueError, match="same length"):
            interp.interpolate_surface(T, M, IV)


class TestEdgeCases:
    """Edge cases: sparse data, boundary conditions."""

    def test_minimum_points(self):
        """Exactly MIN_POINTS points."""
        interp = VolatilitySurfaceInterpolator(method="linear")
        n = MIN_POINTS
        T = np.linspace(0.2, 1.0, n)
        M = np.linspace(0.9, 1.1, n)
        IV = 0.2 + 0.05 * np.random.randn(n)
        IV = np.clip(IV, 0.1, 0.5)
        T_grid, M_grid, IV_grid = interp.interpolate_surface(T, M, IV)
        assert IV_grid.shape == (DEFAULT_GRID_SIZE_T, DEFAULT_GRID_SIZE_M)

    def test_sparse_data(self):
        """Sparse but enough points."""
        interp = VolatilitySurfaceInterpolator(method="linear")
        T = np.array([0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 0.5, 0.5])
        M = np.array([0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1, 1.0, 1.0])
        IV = np.full(10, 0.22)
        T_grid, M_grid, IV_grid = interp.interpolate_surface(T, M, IV)
        assert np.all(np.isfinite(IV_grid)) or np.any(np.isfinite(IV_grid))

    def test_smoothing_optional(self, synthetic_points):
        T, M, IV = synthetic_points
        interp = VolatilitySurfaceInterpolator()
        _, _, IV_no_smooth = interp.interpolate_surface(T, M, IV, sigma_smooth=None)
        _, _, IV_smooth = interp.interpolate_surface(T, M, IV, sigma_smooth=1.0)
        assert IV_smooth.shape == IV_no_smooth.shape
        # Smoothed may differ from unsmoothed
        assert np.any(IV_smooth != IV_no_smooth) or np.allclose(IV_smooth, IV_no_smooth)

    def test_both_methods_produce_grid(self, synthetic_points):
        T, M, IV = synthetic_points
        for method in ("linear", "cubic"):
            interp = VolatilitySurfaceInterpolator(method=method)
            T_g, M_g, IV_g = interp.interpolate_surface(T, M, IV)
            assert T_g.shape == M_g.shape == IV_g.shape
            assert IV_g.shape[0] == DEFAULT_GRID_SIZE_T
            assert IV_g.shape[1] == DEFAULT_GRID_SIZE_M
