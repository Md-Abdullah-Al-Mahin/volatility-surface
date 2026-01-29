"""
Unit tests for Module 6: SurfaceAnalytics
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.analytics import (
    SurfaceAnalytics,
    get_surface_summary,
    compare_surfaces,
)


def _sample_grids(n_t: int = 20, n_m: int = 25):
    """Create sample T_grid, M_grid, IV_grid of shape (n_t, n_m)."""
    T_1d = np.linspace(0.1, 1.0, n_t)
    M_1d = np.linspace(0.9, 1.1, n_m)
    T_grid, M_grid = np.meshgrid(T_1d, M_1d, indexing="ij")
    IV_grid = 0.2 + 0.05 * (M_grid - 1.0) ** 2 + 0.02 * T_grid
    return T_grid, M_grid, IV_grid


class TestGetSurfaceSummary:
    """Tests for get_surface_summary."""

    def test_returns_dict(self):
        T, M, IV = _sample_grids()
        out = get_surface_summary(T, M, IV)
        assert "min_iv" in out and "max_iv" in out and "mean_iv" in out
        assert "n_valid" in out and "T_range" in out and "M_range" in out

    def test_valid_range(self):
        T, M, IV = _sample_grids()
        out = get_surface_summary(T, M, IV)
        assert out["min_iv"] <= out["mean_iv"] <= out["max_iv"]
        assert out["n_valid"] == T.size


class TestSurfaceAnalyticsInit:
    """Tests for SurfaceAnalytics __init__."""

    def test_init_accepts_grids(self):
        T, M, IV = _sample_grids()
        a = SurfaceAnalytics(T, M, IV)
        assert a.T_grid.shape == T.shape
        assert a.IV_grid.shape == IV.shape

    def test_mismatched_shapes_raises(self):
        T, M, IV = _sample_grids(5, 5)
        IV_bad = IV[:3, :]
        with pytest.raises(ValueError, match="same shape"):
            SurfaceAnalytics(T, M, IV_bad)


class TestCalculateSkew:
    """Tests for calculate_skew."""

    def test_returns_metadata(self):
        T, M, IV = _sample_grids()
        a = SurfaceAnalytics(T, M, IV)
        out = a.calculate_skew(target_time=0.5, put_moneyness=0.95, call_moneyness=1.05)
        assert "skew" in out and "iv_put" in out and "iv_call" in out
        assert "target_time" in out and "T_used" in out

    def test_skew_put_minus_call(self):
        T, M, IV = _sample_grids()
        a = SurfaceAnalytics(T, M, IV)
        out = a.calculate_skew(target_time=0.5, put_moneyness=0.95, call_moneyness=1.05)
        expected = out["iv_put"] - out["iv_call"]
        assert out["skew"] == pytest.approx(expected) or (np.isnan(out["skew"]) and np.isnan(expected))


class TestGetTermStructure:
    """Tests for get_term_structure."""

    def test_returns_two_arrays(self):
        T, M, IV = _sample_grids()
        a = SurfaceAnalytics(T, M, IV)
        times, ivs = a.get_term_structure(target_moneyness=1.0)
        assert times.shape == (T.shape[0],)
        assert ivs.shape == (IV.shape[0],)

    def test_same_length(self):
        T, M, IV = _sample_grids(15, 20)
        a = SurfaceAnalytics(T, M, IV)
        times, ivs = a.get_term_structure(target_moneyness=1.0)
        assert len(times) == len(ivs)


class TestCheckCalendarArbitrage:
    """Tests for check_calendar_arbitrage."""

    def test_returns_tuple(self):
        T, M, IV = _sample_grids()
        a = SurfaceAnalytics(T, M, IV)
        is_valid, violations = a.check_calendar_arbitrage(target_moneyness=1.0)
        assert isinstance(is_valid, bool)
        assert isinstance(violations, list)

    def test_violation_structure(self):
        # Build surface where IV*sqrt(T) decreases (arbitrage)
        n_t, n_m = 10, 5
        T_1d = np.linspace(0.2, 1.0, n_t)
        M_1d = np.array([0.98, 1.0, 1.02, 1.05, 1.08])
        T_grid, M_grid = np.meshgrid(T_1d, M_1d, indexing="ij")
        # IV decreasing in T strongly so that IV*sqrt(T) decreases
        IV_grid = 0.5 / np.sqrt(T_grid)  # IV = 0.5/sqrt(T) => IV*sqrt(T)=0.5 constant
        IV_grid = IV_grid + 0.1 * np.random.randn(*IV_grid.shape)  # add noise to get violations
        a = SurfaceAnalytics(T_grid, M_grid, IV_grid)
        is_valid, violations = a.check_calendar_arbitrage(target_moneyness=1.0, tolerance=0.0)
        for v in violations:
            assert "T" in v and "IV" in v and "value" in v and "prev_value" in v


class TestGenerateMetricsReport:
    """Tests for generate_metrics_report."""

    def test_returns_report_dict(self):
        T, M, IV = _sample_grids()
        a = SurfaceAnalytics(T, M, IV)
        report = a.generate_metrics_report()
        assert "skews" in report and "term_structure" in report
        assert "calendar_arbitrage" in report and "summary" in report

    def test_export_csv(self):
        T, M, IV = _sample_grids()
        a = SurfaceAnalytics(T, M, IV)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            report = a.generate_metrics_report(output_path=path)
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "skew" in content or "metric" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestCompareSurfaces:
    """Tests for compare_surfaces."""

    def test_same_shape_returns_report(self):
        T, M, IV1 = _sample_grids(10, 10)
        IV2 = IV1 + 0.01 * np.random.randn(*IV1.shape)
        out = compare_surfaces(T, M, IV1, T, M, IV2, "A", "B")
        assert "diff_grid" in out and "mean_diff" in out
        assert "max_diff" in out and "min_diff" in out
        assert out["label1"] == "A" and out["label2"] == "B"

    def test_diff_grid_shape(self):
        T, M, IV1 = _sample_grids(5, 5)
        IV2 = IV1 + 0.05
        out = compare_surfaces(T, M, IV1, T, M, IV2)
        assert out["diff_grid"].shape == IV1.shape
        assert np.nanmean(out["diff_grid"]) == pytest.approx(0.05)

    def test_different_shapes_raises(self):
        T1, M1, IV1 = _sample_grids(5, 5)
        T2, M2, IV2 = _sample_grids(10, 10)
        with pytest.raises(ValueError, match="same shape"):
            compare_surfaces(T1, M1, IV1, T2, M2, IV2)
