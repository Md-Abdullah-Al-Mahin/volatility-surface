"""
Module 3: SurfaceCoordinateEngine

Transforms raw option data into a normalized coordinate system (T, M, IV)
suitable for 3D volatility surface modeling.
"""

import logging
from datetime import date
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SurfaceCoordinateEngine:
    """
    Transforms options DataFrames (with strike, IV, expiration) into
    (T, M, IV) coordinate arrays: time to expiry, moneyness, implied volatility.
    """

    def __init__(self, moneyness_method: str = "ratio"):
        """
        Initialize the coordinate engine.

        Args:
            moneyness_method: 'ratio' (M = strike/spot) or 'log' (M = log(strike/spot)).
        """
        if moneyness_method not in ("ratio", "log"):
            raise ValueError(
                f"moneyness_method must be 'ratio' or 'log', got {moneyness_method!r}"
            )
        self.moneyness_method = moneyness_method
        logger.info(f"Initialized SurfaceCoordinateEngine with moneyness_method={moneyness_method!r}")

    def _calculate_time_to_expiry(
        self,
        exp_date: Union[date, str, pd.Timestamp],
        current_date: Optional[Union[date, str, pd.Timestamp]] = None,
    ) -> float:
        """
        Convert expiration and current date to time to expiry in years.

        Args:
            exp_date: Expiration date.
            current_date: Reference date (default: today).

        Returns:
            Annualized fraction: (exp_date - current_date).days / 365.0.
            Non-positive for same day or past dates.
        """
        if current_date is None:
            current_date = date.today()
        exp = self._to_date(exp_date)
        curr = self._to_date(current_date)
        days = (exp - curr).days
        return days / 365.0

    @staticmethod
    def _to_date(
        value: Union[date, str, pd.Timestamp],
    ) -> date:
        """Convert various date types to datetime.date."""
        if isinstance(value, date) and not isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, pd.Timestamp):
            return value.date()
        if isinstance(value, str):
            return date.fromisoformat(value[:10])
        raise TypeError(f"Expected date, str, or Timestamp, got {type(value)}")

    def _calculate_moneyness(
        self,
        strike: float,
        spot: float,
        method: Optional[str] = None,
    ) -> float:
        """
        Compute moneyness for a strike and spot price.

        Args:
            strike: Strike price.
            spot: Spot (underlying) price.
            method: Override instance method: 'ratio' or 'log'.

        Returns:
            Ratio method: M = strike / spot.
            Log method: M = log(strike / spot).
        """
        if spot <= 0:
            return np.nan
        method = method or self.moneyness_method
        if method == "ratio":
            return float(strike / spot)
        if method == "log":
            ratio = strike / spot
            if ratio <= 0:
                return np.nan
            return float(np.log(ratio))
        raise ValueError(f"moneyness method must be 'ratio' or 'log', got {method!r}")

    def _filter_extreme_moneyness(
        self,
        df: pd.DataFrame,
        min_m: float = 0.7,
        max_m: float = 1.3,
    ) -> pd.DataFrame:
        """
        Keep only rows where min_m <= moneyness <= max_m.

        Args:
            df: DataFrame with column 'M' (moneyness), and optionally 'T', 'IV'.
            min_m: Lower moneyness bound.
            max_m: Upper moneyness bound.

        Returns:
            Filtered DataFrame.
        """
        if "M" not in df.columns:
            logger.warning("DataFrame has no 'M' column, returning unchanged")
            return df
        mask = (df["M"] >= min_m) & (df["M"] <= max_m)
        n_before = len(df)
        filtered = df.loc[mask]
        n_after = len(filtered)
        n_removed = n_before - n_after
        if n_removed > 0:
            logger.info(
                f"Filtered {n_removed} rows outside moneyness [{min_m}, {max_m}] "
                f"({n_before} -> {n_after})"
            )
        return filtered

    def transform_to_coordinates(
        self,
        dataframes_dict: Dict[str, pd.DataFrame],
        spot_price: float,
        moneyness_method: Optional[str] = None,
        filter_extremes: bool = True,
        min_m: float = 0.7,
        max_m: float = 1.3,
        min_time_to_expiry: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build (T, M, IV) arrays from options DataFrames.

        Args:
            dataframes_dict: Map expiration date string -> DataFrame with
                columns: strike, impliedVolatility, expirationDate (or use key).
            spot_price: Current underlying price.
            moneyness_method: Override instance method ('ratio' or 'log').
            filter_extremes: If True, drop points outside [min_m, max_m].
            min_m: Lower moneyness bound when filtering.
            max_m: Upper moneyness bound when filtering.
            min_time_to_expiry: Skip expirations with T < this (years).

        Returns:
            (T, M, IV) as 1D numpy arrays of same length.

        Raises:
            ValueError: If result is empty or has too few points.
        """
        method = moneyness_method or self.moneyness_method
        current_date = date.today()
        T_list: list = []
        M_list: list = []
        IV_list: list = []

        for exp_key, df in dataframes_dict.items():
            if df is None or df.empty:
                continue
            if "strike" not in df.columns or "impliedVolatility" not in df.columns:
                logger.warning(
                    f"DataFrame for {exp_key} missing strike or impliedVolatility, skipping"
                )
                continue

            # Time to expiry for this expiration
            try:
                if "expirationDate" in df.columns and df["expirationDate"].notna().any():
                    exp_val = df["expirationDate"].iloc[0]
                    exp_d = self._to_date(exp_val)
                else:
                    exp_d = self._to_date(exp_key)
            except (TypeError, ValueError):
                try:
                    exp_d = self._to_date(exp_key)
                except (TypeError, ValueError):
                    logger.warning(f"Could not parse expiration for {exp_key}, skipping")
                    continue

            T_years = self._calculate_time_to_expiry(exp_d, current_date)
            if T_years < min_time_to_expiry:
                logger.debug(f"Skipping expiration {exp_key}: T={T_years} < min")
                continue

            for _, row in df.iterrows():
                iv = row.get("impliedVolatility")
                if pd.isna(iv) or (isinstance(iv, (int, float)) and iv < 0):
                    continue
                strike = row.get("strike")
                if pd.isna(strike) or strike <= 0:
                    continue
                m = self._calculate_moneyness(float(strike), float(spot_price), method)
                if np.isnan(m):
                    continue
                T_list.append(T_years)
                M_list.append(m)
                IV_list.append(float(iv))

        coord_df = pd.DataFrame({"T": T_list, "M": M_list, "IV": IV_list})

        if filter_extremes and len(coord_df) > 0:
            coord_df = self._filter_extreme_moneyness(coord_df, min_m=min_m, max_m=max_m)

        T_arr = coord_df["T"].values
        M_arr = coord_df["M"].values
        IV_arr = coord_df["IV"].values

        # Validation
        if len(T_arr) == 0:
            raise ValueError(
                "No valid (T, M, IV) points: check data has strike, impliedVolatility, "
                "and valid expirations."
            )
        min_points = 10
        if len(T_arr) < min_points:
            logger.warning(
                f"Only {len(T_arr)} points after transform (recommend at least {min_points})"
            )

        logger.info(
            f"transform_to_coordinates: {len(T_arr)} points, "
            f"T in [{T_arr.min():.4f}, {T_arr.max():.4f}], "
            f"M in [{M_arr.min():.4f}, {M_arr.max():.4f}]"
        )
        return T_arr, M_arr, IV_arr
