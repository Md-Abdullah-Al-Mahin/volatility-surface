"""
Unit tests for Module 3: SurfaceCoordinateEngine
"""

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.coordinate_engine import SurfaceCoordinateEngine


class TestSurfaceCoordinateEngineInit:
    """Tests for __init__."""

    def test_init_default(self):
        engine = SurfaceCoordinateEngine()
        assert engine.moneyness_method == "ratio"

    def test_init_ratio(self):
        engine = SurfaceCoordinateEngine(moneyness_method="ratio")
        assert engine.moneyness_method == "ratio"

    def test_init_log(self):
        engine = SurfaceCoordinateEngine(moneyness_method="log")
        assert engine.moneyness_method == "log"

    def test_init_invalid_raises(self):
        with pytest.raises(ValueError, match="moneyness_method must be 'ratio' or 'log'"):
            SurfaceCoordinateEngine(moneyness_method="invalid")


class TestToDate:
    """Tests for _to_date helper."""

    def test_date_unchanged(self):
        d = date(2024, 6, 15)
        assert SurfaceCoordinateEngine._to_date(d) == d

    def test_iso_string(self):
        assert SurfaceCoordinateEngine._to_date("2024-06-15") == date(2024, 6, 15)

    def test_timestamp(self):
        ts = pd.Timestamp("2024-06-15")
        assert SurfaceCoordinateEngine._to_date(ts) == date(2024, 6, 15)


class TestCalculateTimeToExpiry:
    """Tests for _calculate_time_to_expiry."""

    def test_future_expiry(self):
        engine = SurfaceCoordinateEngine()
        exp = date.today() + timedelta(days=365)
        T = engine._calculate_time_to_expiry(exp, date.today())
        assert abs(T - 1.0) < 0.01

    def test_same_day(self):
        engine = SurfaceCoordinateEngine()
        today = date(2024, 6, 15)
        T = engine._calculate_time_to_expiry(today, today)
        assert T == 0.0

    def test_past_date(self):
        engine = SurfaceCoordinateEngine()
        past = date.today() - timedelta(days=30)
        T = engine._calculate_time_to_expiry(past, date.today())
        assert T < 0

    def test_half_year(self):
        engine = SurfaceCoordinateEngine()
        exp = date(2024, 6, 15)
        curr = date(2023, 12, 15)
        T = engine._calculate_time_to_expiry(exp, curr)
        assert abs(T - 0.5) < 0.01

    def test_uses_today_when_current_none(self):
        engine = SurfaceCoordinateEngine()
        exp = date.today() + timedelta(days=90)
        T = engine._calculate_time_to_expiry(exp)
        assert T > 0
        assert abs(T - 90 / 365.0) < 0.01


class TestCalculateMoneyness:
    """Tests for _calculate_moneyness."""

    def test_ratio_atm(self):
        engine = SurfaceCoordinateEngine(moneyness_method="ratio")
        m = engine._calculate_moneyness(100.0, 100.0)
        assert m == pytest.approx(1.0)

    def test_ratio_itm_call(self):
        engine = SurfaceCoordinateEngine(moneyness_method="ratio")
        m = engine._calculate_moneyness(90.0, 100.0)
        assert m == pytest.approx(0.9)

    def test_ratio_otm_call(self):
        engine = SurfaceCoordinateEngine(moneyness_method="ratio")
        m = engine._calculate_moneyness(110.0, 100.0)
        assert m == pytest.approx(1.1)

    def test_log_atm(self):
        engine = SurfaceCoordinateEngine(moneyness_method="log")
        m = engine._calculate_moneyness(100.0, 100.0)
        assert m == pytest.approx(0.0)

    def test_log_otm(self):
        engine = SurfaceCoordinateEngine(moneyness_method="log")
        m = engine._calculate_moneyness(110.0, 100.0)
        assert m == pytest.approx(np.log(1.1))

    def test_override_method(self):
        engine = SurfaceCoordinateEngine(moneyness_method="ratio")
        m = engine._calculate_moneyness(100.0, 100.0, method="log")
        assert m == pytest.approx(0.0)

    def test_zero_spot_returns_nan(self):
        engine = SurfaceCoordinateEngine(moneyness_method="ratio")
        m = engine._calculate_moneyness(100.0, 0.0)
        assert np.isnan(m)


class TestFilterExtremeMoneyness:
    """Tests for _filter_extreme_moneyness."""

    def test_keeps_in_bounds(self):
        engine = SurfaceCoordinateEngine()
        df = pd.DataFrame({"T": [0.5, 0.5], "M": [0.9, 1.1], "IV": [0.2, 0.22]})
        out = engine._filter_extreme_moneyness(df, min_m=0.7, max_m=1.3)
        assert len(out) == 2
        pd.testing.assert_frame_equal(out.reset_index(drop=True), df)

    def test_removes_outside_bounds(self):
        engine = SurfaceCoordinateEngine()
        df = pd.DataFrame({
            "T": [0.5, 0.5, 0.5],
            "M": [0.5, 1.0, 1.5],
            "IV": [0.3, 0.2, 0.25],
        })
        out = engine._filter_extreme_moneyness(df, min_m=0.7, max_m=1.3)
        assert len(out) == 1
        assert out["M"].iloc[0] == pytest.approx(1.0)

    def test_empty_df(self):
        engine = SurfaceCoordinateEngine()
        df = pd.DataFrame({"T": [], "M": [], "IV": []})
        out = engine._filter_extreme_moneyness(df, min_m=0.7, max_m=1.3)
        assert len(out) == 0


class TestTransformToCoordinates:
    """Tests for transform_to_coordinates."""

    @pytest.fixture
    def engine(self):
        return SurfaceCoordinateEngine(moneyness_method="ratio")

    @pytest.fixture
    def sample_dataframes(self):
        """DataFrames with strike, impliedVolatility, expirationDate."""
        exp1 = (date.today() + timedelta(days=30)).isoformat()
        exp2 = (date.today() + timedelta(days=90)).isoformat()
        df1 = pd.DataFrame({
            "strike": [95.0, 100.0, 105.0],
            "impliedVolatility": [0.22, 0.20, 0.21],
            "expirationDate": [exp1, exp1, exp1],
        })
        df2 = pd.DataFrame({
            "strike": [98.0, 100.0, 102.0],
            "impliedVolatility": [0.23, 0.21, 0.22],
            "expirationDate": [exp2, exp2, exp2],
        })
        return {exp1: df1, exp2: df2}

    def test_returns_three_arrays(self, engine, sample_dataframes):
        T, M, IV = engine.transform_to_coordinates(
            sample_dataframes, spot_price=100.0, filter_extremes=False
        )
        assert isinstance(T, np.ndarray)
        assert isinstance(M, np.ndarray)
        assert isinstance(IV, np.ndarray)
        assert len(T) == len(M) == len(IV)
        assert len(T) == 6  # 3 + 3 rows

    def test_moneyness_ratio(self, engine, sample_dataframes):
        T, M, IV = engine.transform_to_coordinates(
            sample_dataframes, spot_price=100.0, filter_extremes=False
        )
        # Strikes 95, 100, 105, 98, 100, 102 -> M = 0.95, 1.0, 1.05, 0.98, 1.0, 1.02
        assert 0.9 < M.min() < 1.0
        assert 1.0 < M.max() < 1.1

    def test_filter_extremes(self, engine, sample_dataframes):
        # Add an extreme strike
        exp = list(sample_dataframes.keys())[0]
        sample_dataframes[exp] = pd.concat([
            sample_dataframes[exp],
            pd.DataFrame({
                "strike": [50.0],
                "impliedVolatility": [0.5],
                "expirationDate": [exp],
            }),
        ], ignore_index=True)
        T, M, IV = engine.transform_to_coordinates(
            sample_dataframes, spot_price=100.0, filter_extremes=True,
            min_m=0.7, max_m=1.3,
        )
        # M=0.5 should be filtered out
        assert np.all(M >= 0.7)
        assert np.all(M <= 1.3)

    def test_skips_nan_iv(self, engine, sample_dataframes):
        exp = list(sample_dataframes.keys())[0]
        sample_dataframes[exp].loc[0, "impliedVolatility"] = np.nan
        T, M, IV = engine.transform_to_coordinates(
            sample_dataframes, spot_price=100.0, filter_extremes=False
        )
        assert len(T) == 5  # one row skipped

    def test_empty_dict_raises(self, engine):
        with pytest.raises(ValueError, match="No valid"):
            engine.transform_to_coordinates({}, spot_price=100.0)

    def test_empty_dataframes_raises(self, engine):
        exp = (date.today() + timedelta(days=30)).isoformat()
        with pytest.raises(ValueError, match="No valid"):
            engine.transform_to_coordinates({exp: pd.DataFrame()}, spot_price=100.0)

    def test_uses_exp_key_when_no_column(self, engine):
        exp_key = (date.today() + timedelta(days=60)).isoformat()
        df = pd.DataFrame({
            "strike": [100.0],
            "impliedVolatility": [0.2],
            # no expirationDate
        })
        T, M, IV = engine.transform_to_coordinates(
            {exp_key: df}, spot_price=100.0, filter_extremes=False
        )
        assert len(T) == 1
        assert T[0] > 0

    def test_log_moneyness(self, sample_dataframes):
        engine = SurfaceCoordinateEngine(moneyness_method="log")
        T, M, IV = engine.transform_to_coordinates(
            sample_dataframes, spot_price=100.0, filter_extremes=False
        )
        # ATM strike 100 -> log(1)=0
        assert 0.0 in M or np.isclose(M, 0.0).any()
        assert M.min() < 0  # strikes below spot
        assert M.max() > 0  # strikes above spot


class TestEdgeCases:
    """Edge cases."""

    def test_single_expiration(self):
        engine = SurfaceCoordinateEngine()
        exp = (date.today() + timedelta(days=45)).isoformat()
        df = pd.DataFrame({
            "strike": [99.0, 100.0, 101.0],
            "impliedVolatility": [0.21, 0.20, 0.21],
            "expirationDate": [exp, exp, exp],
        })
        T, M, IV = engine.transform_to_coordinates(
            {exp: df}, spot_price=100.0, filter_extremes=False
        )
        assert len(T) == 3
        assert np.allclose(T, T[0])
        assert len(np.unique(M)) == 3

    def test_missing_strike_column_skips_df(self):
        engine = SurfaceCoordinateEngine()
        exp = (date.today() + timedelta(days=30)).isoformat()
        df = pd.DataFrame({
            "impliedVolatility": [0.2],
            "expirationDate": [exp],
        })
        with pytest.raises(ValueError, match="No valid"):
            engine.transform_to_coordinates({exp: df}, spot_price=100.0)
