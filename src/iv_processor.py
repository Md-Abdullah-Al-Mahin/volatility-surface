"""
Module 2: ImpliedVolatilityProcessor

Core reliability module ensuring a complete, accurate set of implied volatilities.
Uses Black-Scholes and Newton-Raphson (with bisection fallback) to compute IVs.
"""

import logging
from datetime import date
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

logger = logging.getLogger(__name__)


class ImpliedVolatilityProcessor:
    """
    Processes options data to ensure complete, validated implied volatilities.
    Fetches risk-free rate from ^TNX or uses a constant; computes missing IVs
    via Black-Scholes and Newton-Raphson (or bisection).
    """

    def __init__(
        self,
        risk_free_rate: Optional[float] = None,
        use_treasury_rate: bool = True,
    ):
        """
        Initialize the IV processor.

        Args:
            risk_free_rate: Fixed risk-free rate (decimal, e.g. 0.05 for 5%).
                           Used when use_treasury_rate is False or fetch fails.
            use_treasury_rate: If True, try to fetch 10Y Treasury (^TNX) rate;
                              fall back to risk_free_rate or 0.05.
        """
        self._risk_free_rate_constant = risk_free_rate if risk_free_rate is not None else 0.05
        self.use_treasury_rate = use_treasury_rate
        logger.info(
            f"Initialized ImpliedVolatilityProcessor use_treasury_rate={use_treasury_rate}, "
            f"fallback_r={self._risk_free_rate_constant}"
        )

    def _get_risk_free_rate(self) -> float:
        """
        Get risk-free rate: from ^TNX if enabled, else constant.

        Returns:
            Risk-free rate as decimal (e.g. 0.05 for 5%).
        """
        if self.use_treasury_rate:
            try:
                tnx = yf.Ticker("^TNX")
                info = tnx.fast_info
                # fast_info can have 'last_price' or we may need history
                if hasattr(info, "last_price") and info.last_price is not None:
                    rate = float(info.last_price) / 100.0
                    logger.debug(f"Using ^TNX rate: {rate:.4f}")
                    return rate
                hist = tnx.history(period="5d")
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    rate = float(hist["Close"].iloc[-1]) / 100.0
                    logger.debug(f"Using ^TNX rate from history: {rate:.4f}")
                    return rate
            except Exception as e:
                logger.warning(f"Could not fetch ^TNX rate: {e}, using fallback")
        rate = self._risk_free_rate_constant
        logger.debug(f"Using fallback risk-free rate: {rate:.4f}")
        return rate

    def _calculate_consensus_price(self, row: pd.Series) -> float:
        """
        Consensus price for an option: mid (bid+ask)/2 preferred, else lastPrice.

        Args:
            row: Option row with bid, ask, lastPrice.

        Returns:
            Consensus price, or np.nan if neither available.
        """
        bid = row.get("bid")
        ask = row.get("ask")
        if pd.notna(bid) and pd.notna(ask):
            return float(bid + ask) / 2.0
        last = row.get("lastPrice")
        if pd.notna(last):
            return float(last)
        return np.nan

    def _black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """
        Black-Scholes option price (European).

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to expiry (years).
            r: Risk-free rate (decimal).
            sigma: Volatility (decimal).
            option_type: 'call' or 'put'.

        Returns:
            Option price.
        """
        if T <= 0 or sigma <= 0:
            return np.nan
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        if option_type.lower() == "call":
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    def _black_scholes_vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Vega = S * sqrt(T) * n(d1). Used in Newton-Raphson."""
        if T <= 0 or sigma <= 0:
            return 0.0
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        return float(S * sqrt_T * norm.pdf(d1))

    def _calculate_iv_newton_raphson(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        market_price: float,
        option_type: str,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Optional[float]:
        """
        Solve for IV such that BS_price(IV) = market_price using Newton-Raphson.

        Args:
            S, K, T, r: Spot, strike, time (years), rate.
            market_price: Observed option price.
            option_type: 'call' or 'put'.
            max_iter: Maximum iterations.
            tol: Convergence tolerance on price.

        Returns:
            Implied volatility (decimal), or None if failed.
        """
        if T <= 0 or market_price <= 0:
            return None
        # Initial guess: 0.2 (20%)
        sigma = 0.2
        for _ in range(max_iter):
            price = self._black_scholes_price(S, K, T, r, sigma, option_type)
            if np.isnan(price):
                return None
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            vega = self._black_scholes_vega(S, K, T, r, sigma)
            if vega < 1e-10:
                break
            sigma = sigma - diff / vega
            if sigma <= 0:
                sigma = 1e-6
            if sigma > 5.0:
                sigma = 0.5
        # Fallback: bisection
        return self._calculate_iv_bisection(S, K, T, r, market_price, option_type, max_iter=max_iter, tol=tol)

    def _calculate_iv_bisection(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        market_price: float,
        option_type: str,
        low: float = 1e-4,
        high: float = 3.0,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Optional[float]:
        """
        Solve for IV using bisection (fallback when Newton-Raphson fails).
        """
        if T <= 0 or market_price <= 0:
            return None
        plow = self._black_scholes_price(S, K, T, r, low, option_type)
        phigh = self._black_scholes_price(S, K, T, r, high, option_type)
        if np.isnan(plow) or np.isnan(phigh):
            return None
        if plow > market_price or phigh < market_price:
            return None
        for _ in range(max_iter):
            mid = (low + high) / 2
            pmid = self._black_scholes_price(S, K, T, r, mid, option_type)
            if np.isnan(pmid) or abs(pmid - market_price) < tol:
                return mid
            if pmid > market_price:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def _validate_iv(self, iv_value: float) -> bool:
        """
        Check if IV is in acceptable range: non-negative and <= 2.0 (200%).

        Args:
            iv_value: Implied volatility (decimal).

        Returns:
            True if valid.
        """
        if pd.isna(iv_value) or iv_value < 0:
            return False
        if iv_value > 2.0:
            return False
        return True

    def process_iv(
        self,
        dataframes_dict: Dict[str, pd.DataFrame],
        spot_price: float,
    ) -> Dict[str, pd.DataFrame]:
        """
        Ensure every option row has a valid impliedVolatility; compute missing ones.

        Args:
            dataframes_dict: Map expiration date string -> DataFrame (from data fetcher).
            spot_price: Current underlying price (used for all expirations).

        Returns:
            Same structure with impliedVolatility column added/updated.
        """
        r = self._get_risk_free_rate()
        result = {}
        total_used = 0
        total_calculated = 0
        total_invalid = 0

        for exp_date, df in dataframes_dict.items():
            if df is None or df.empty:
                result[exp_date] = df
                continue
            df = df.copy()
            if "impliedVolatility" not in df.columns:
                df["impliedVolatility"] = np.nan

            # Time to expiry (years) for this expiration
            try:
                if "expirationDate" in df.columns and df["expirationDate"].notna().any():
                    exp_str = df["expirationDate"].iloc[0]
                    if isinstance(exp_str, str):
                        exp_d = date.fromisoformat(exp_str[:10])
                    else:
                        exp_d = pd.Timestamp(exp_str).date()
                    today = date.today()
                    T_days = (exp_d - today).days
                    T_annual = max(T_days / 365.0, 1e-6)
                else:
                    # Use key exp_date if it looks like YYYY-MM-DD
                    try:
                        exp_d = date.fromisoformat(exp_date[:10])
                        today = date.today()
                        T_annual = max((exp_d - today).days / 365.0, 1e-6)
                    except Exception:
                        T_annual = 1.0 / 12.0
            except Exception:
                T_annual = 1.0 / 12.0

            ivs = []
            for idx, row in df.iterrows():
                consensus = self._calculate_consensus_price(row)
                strike = row.get("strike")
                opt_type = "call" if str(row.get("optionType", "call")).lower() == "call" else "put"
                existing_iv = row.get("impliedVolatility")

                if self._validate_iv(existing_iv):
                    ivs.append(float(existing_iv))
                    total_used += 1
                    continue
                if pd.isna(consensus) or consensus <= 0 or pd.isna(strike) or strike <= 0:
                    ivs.append(np.nan)
                    total_invalid += 1
                    continue
                S, K = float(spot_price), float(strike)
                iv = self._calculate_iv_newton_raphson(S, K, T_annual, r, consensus, opt_type)
                if iv is not None and self._validate_iv(iv):
                    ivs.append(iv)
                    total_calculated += 1
                else:
                    ivs.append(np.nan)
                    total_invalid += 1

            df["impliedVolatility"] = ivs
            result[exp_date] = df

        logger.info(
            f"process_iv: used existing IV={total_used}, calculated={total_calculated}, invalid/missing={total_invalid}"
        )
        return result
