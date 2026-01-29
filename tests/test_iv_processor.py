"""
Unit tests for Module 2: ImpliedVolatilityProcessor
"""

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.iv_processor import ImpliedVolatilityProcessor


class TestImpliedVolatilityProcessorInit:
    """Tests for __init__ and risk-free rate source."""

    def test_init_default(self):
        processor = ImpliedVolatilityProcessor()
        assert processor._risk_free_rate_constant == 0.05
        assert processor.use_treasury_rate is True

    def test_init_custom_rate_no_treasury(self):
        processor = ImpliedVolatilityProcessor(risk_free_rate=0.03, use_treasury_rate=False)
        assert processor._risk_free_rate_constant == 0.03
        assert processor.use_treasury_rate is False

    def test_init_custom_rate_fallback(self):
        processor = ImpliedVolatilityProcessor(risk_free_rate=0.04)
        assert processor._risk_free_rate_constant == 0.04


class TestGetRiskFreeRate:
    """Tests for _get_risk_free_rate."""

    def test_uses_constant_when_disabled(self):
        processor = ImpliedVolatilityProcessor(risk_free_rate=0.06, use_treasury_rate=False)
        assert processor._get_risk_free_rate() == 0.06

    @patch("src.iv_processor.yf.Ticker")
    def test_uses_treasury_when_available(self, mock_ticker_class):
        mock_ticker = mock_ticker_class.return_value
        mock_fast_info = type("FastInfo", (), {"last_price": 4.5})()
        mock_ticker.fast_info = mock_fast_info  # 4.5% yield
        processor = ImpliedVolatilityProcessor(risk_free_rate=0.05, use_treasury_rate=True)
        assert processor._get_risk_free_rate() == pytest.approx(0.045)

    @patch("src.iv_processor.yf.Ticker")
    def test_fallback_on_treasury_failure(self, mock_ticker_class):
        mock_ticker_class.side_effect = Exception("network error")
        processor = ImpliedVolatilityProcessor(risk_free_rate=0.04, use_treasury_rate=True)
        assert processor._get_risk_free_rate() == 0.04


class TestConsensusPrice:
    """Tests for _calculate_consensus_price."""

    def test_bid_ask_mid(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        row = pd.Series({"bid": 10.0, "ask": 12.0, "lastPrice": 11.5})
        assert processor._calculate_consensus_price(row) == 11.0

    def test_last_price_fallback(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        row = pd.Series({"bid": np.nan, "ask": np.nan, "lastPrice": 9.0})
        assert processor._calculate_consensus_price(row) == 9.0

    def test_nan_when_missing(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        row = pd.Series({"bid": np.nan, "ask": np.nan, "lastPrice": np.nan})
        result = processor._calculate_consensus_price(row)
        assert pd.isna(result)


class TestBlackScholes:
    """Tests for Black-Scholes pricing (known values)."""

    def test_call_price_known(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        # S=100, K=100, T=1, r=0.05, sigma=0.2 -> known approx
        price = processor._black_scholes_price(100, 100, 1.0, 0.05, 0.2, "call")
        assert price > 0
        # Call ~10.45 for ATM 1Y with sigma=0.2
        assert 9 < price < 12

    def test_put_price_known(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        price = processor._black_scholes_price(100, 100, 1.0, 0.05, 0.2, "put")
        assert price > 0
        assert 5 < price < 8

    def test_put_call_parity_approx(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.25
        C = processor._black_scholes_price(S, K, T, r, sigma, "call")
        P = processor._black_scholes_price(S, K, T, r, sigma, "put")
        # C - P = S - K*exp(-r*T)
        parity_diff = (C - P) - (S - K * np.exp(-r * T))
        assert abs(parity_diff) < 1e-6

    def test_zero_time_returns_nan(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        price = processor._black_scholes_price(100, 100, 0.0, 0.05, 0.2, "call")
        assert np.isnan(price)


class TestIVSolver:
    """Tests for IV calculation (reverse: price -> IV)."""

    def test_iv_recovers_sigma(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        S, K, T, r = 100, 100, 1.0, 0.05
        target_sigma = 0.25
        market_price = processor._black_scholes_price(S, K, T, r, target_sigma, "call")
        iv = processor._calculate_iv_newton_raphson(S, K, T, r, market_price, "call")
        assert iv is not None
        assert abs(iv - target_sigma) < 0.005  # within 0.5%

    def test_iv_put(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        S, K, T, r = 100, 95, 0.5, 0.05
        target_sigma = 0.20
        market_price = processor._black_scholes_price(S, K, T, r, target_sigma, "put")
        iv = processor._calculate_iv_newton_raphson(S, K, T, r, market_price, "put")
        assert iv is not None
        assert abs(iv - target_sigma) < 0.005

    def test_iv_returns_none_for_zero_price(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        iv = processor._calculate_iv_newton_raphson(100, 100, 1.0, 0.05, 0.0, "call")
        assert iv is None

    def test_iv_returns_none_for_zero_time(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        iv = processor._calculate_iv_newton_raphson(100, 100, 0.0, 0.05, 10.0, "call")
        assert iv is None


class TestValidateIV:
    """Tests for _validate_iv."""

    def test_valid_iv_accepted(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        assert processor._validate_iv(0.2) is True
        assert processor._validate_iv(0.0) is True
        assert processor._validate_iv(2.0) is True

    def test_negative_rejected(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        assert processor._validate_iv(-0.1) is False

    def test_above_200_rejected(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        assert processor._validate_iv(2.5) is False

    def test_nan_rejected(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        assert processor._validate_iv(np.nan) is False


class TestProcessIV:
    """Tests for process_iv main method."""

    @pytest.fixture
    def processor(self):
        return ImpliedVolatilityProcessor(risk_free_rate=0.05, use_treasury_rate=False)

    @pytest.fixture
    def sample_df(self):
        exp = (date.today() + timedelta(days=30)).isoformat()
        return pd.DataFrame({
            "contractSymbol": ["SPY240219C00100000", "SPY240219P00100000"],
            "strike": [100.0, 100.0],
            "lastPrice": [5.0, 2.0],
            "bid": [4.9, 1.9],
            "ask": [5.1, 2.1],
            "optionType": ["call", "put"],
            "expirationDate": [exp, exp],
            "impliedVolatility": [np.nan, np.nan],
        })

    def test_process_iv_adds_iv_column(self, processor, sample_df):
        exp_key = (date.today() + timedelta(days=30)).isoformat()
        data = {exp_key: sample_df}
        result = processor.process_iv(data, spot_price=100.0)
        assert exp_key in result
        df = result[exp_key]
        assert "impliedVolatility" in df.columns
        assert df["impliedVolatility"].notna().any()

    def test_process_iv_keeps_valid_existing_iv(self, processor, sample_df):
        exp_key = (date.today() + timedelta(days=30)).isoformat()
        sample_df["impliedVolatility"] = [0.22, 0.21]
        data = {exp_key: sample_df}
        result = processor.process_iv(data, spot_price=100.0)
        df = result[exp_key]
        assert df["impliedVolatility"].iloc[0] == pytest.approx(0.22)
        assert df["impliedVolatility"].iloc[1] == pytest.approx(0.21)

    def test_process_iv_replaces_invalid_existing_iv(self, processor, sample_df):
        exp_key = (date.today() + timedelta(days=30)).isoformat()
        sample_df["impliedVolatility"] = [2.5, -0.1]  # invalid
        data = {exp_key: sample_df}
        result = processor.process_iv(data, spot_price=100.0)
        df = result[exp_key]
        # Should have been recomputed or NaN
        assert df["impliedVolatility"].iloc[0] <= 2.0 or np.isnan(df["impliedVolatility"].iloc[0])
        assert df["impliedVolatility"].iloc[1] >= 0 or np.isnan(df["impliedVolatility"].iloc[1])

    def test_process_iv_empty_dict(self, processor):
        result = processor.process_iv({}, spot_price=100.0)
        assert result == {}

    def test_process_iv_empty_dataframe(self, processor):
        result = processor.process_iv({"2024-02-19": pd.DataFrame()}, spot_price=100.0)
        assert "2024-02-19" in result
        assert result["2024-02-19"].empty


class TestEdgeCases:
    """Edge cases: very low/high strikes, short/long expiry."""

    def test_short_expiry(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        S, K, T, r = 100, 100, 1.0 / 365, 0.05  # 1 day
        market_price = processor._black_scholes_price(S, K, T, r, 0.3, "call")
        iv = processor._calculate_iv_newton_raphson(S, K, T, r, market_price, "call")
        assert iv is not None
        assert abs(iv - 0.3) < 0.02

    def test_high_strike_otm_call(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        S, K, T, r = 100, 150, 0.5, 0.05
        market_price = processor._black_scholes_price(S, K, T, r, 0.25, "call")
        iv = processor._calculate_iv_newton_raphson(S, K, T, r, market_price, "call")
        assert iv is not None
        assert abs(iv - 0.25) < 0.01

    def test_low_strike_otm_put(self):
        processor = ImpliedVolatilityProcessor(use_treasury_rate=False)
        S, K, T, r = 100, 80, 0.5, 0.05
        market_price = processor._black_scholes_price(S, K, T, r, 0.20, "put")
        iv = processor._calculate_iv_newton_raphson(S, K, T, r, market_price, "put")
        assert iv is not None
        assert abs(iv - 0.20) < 0.01
