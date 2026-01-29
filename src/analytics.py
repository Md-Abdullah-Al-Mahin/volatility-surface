"""
Module 6: SurfaceAnalytics

Extracts quantitative metrics from the constructed surface grid:
skew, term structure, calendar arbitrage checks, surface comparison.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def get_surface_summary(
    T_grid: np.ndarray,
    M_grid: np.ndarray,
    IV_grid: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute a short summary of the volatility surface grid.

    Args:
        T_grid, M_grid, IV_grid: 2D arrays from VolatilitySurfaceInterpolator.

    Returns:
        Dict with min_iv, max_iv, mean_iv, n_valid, T_range, M_range.
    """
    valid = np.isfinite(IV_grid)
    iv_flat = IV_grid[valid]
    if iv_flat.size == 0:
        return {
            "min_iv": None,
            "max_iv": None,
            "mean_iv": None,
            "n_valid": 0,
            "T_range": (float(np.nanmin(T_grid)), float(np.nanmax(T_grid))),
            "M_range": (float(np.nanmin(M_grid)), float(np.nanmax(M_grid))),
        }
    return {
        "min_iv": float(np.min(iv_flat)),
        "max_iv": float(np.max(iv_flat)),
        "mean_iv": float(np.mean(iv_flat)),
        "n_valid": int(valid.sum()),
        "T_range": (float(np.nanmin(T_grid)), float(np.nanmax(T_grid))),
        "M_range": (float(np.nanmin(M_grid)), float(np.nanmax(M_grid))),
    }


class SurfaceAnalytics:
    """
    Analytics on a volatility surface grid (T_grid, M_grid, IV_grid).
    Grid layout: T varies along axis 0, M along axis 1.
    """

    def __init__(
        self,
        T_grid: np.ndarray,
        M_grid: np.ndarray,
        IV_grid: np.ndarray,
    ):
        """
        Initialize with surface grid data from VolatilitySurfaceInterpolator.

        Args:
            T_grid, M_grid, IV_grid: 2D arrays of same shape (n_T, n_M).
        """
        T_grid = np.asarray(T_grid)
        M_grid = np.asarray(M_grid)
        IV_grid = np.asarray(IV_grid)
        if T_grid.shape != M_grid.shape or M_grid.shape != IV_grid.shape:
            raise ValueError(
                f"T_grid, M_grid, IV_grid must have same shape; "
                f"got {T_grid.shape}, {M_grid.shape}, {IV_grid.shape}"
            )
        self.T_grid = T_grid
        self.M_grid = M_grid
        self.IV_grid = IV_grid
        self._T_axis = T_grid[:, 0]
        self._M_axis = M_grid[0, :]

    def calculate_skew(
        self,
        target_time: float,
        put_moneyness: float = 0.95,
        call_moneyness: float = 1.05,
    ) -> Dict[str, Any]:
        """
        Skew at a target time: IV(put_moneyness) - IV(call_moneyness).

        Args:
            target_time: Time to expiry (years).
            put_moneyness: Moneyness for put side (e.g. 0.95).
            call_moneyness: Moneyness for call side (e.g. 1.05).

        Returns:
            Dict with skew, iv_put, iv_call, target_time, put_moneyness, call_moneyness.
        """
        i = np.argmin(np.abs(self._T_axis - target_time))
        j_put = np.argmin(np.abs(self._M_axis - put_moneyness))
        j_call = np.argmin(np.abs(self._M_axis - call_moneyness))
        iv_put = float(self.IV_grid[i, j_put])
        iv_call = float(self.IV_grid[i, j_call])
        if not np.isfinite(iv_put):
            iv_put = np.nan
        if not np.isfinite(iv_call):
            iv_call = np.nan
        skew = float(iv_put - iv_call) if (np.isfinite(iv_put) and np.isfinite(iv_call)) else np.nan
        return {
            "skew": skew,
            "iv_put": iv_put,
            "iv_call": iv_call,
            "target_time": target_time,
            "T_used": float(self._T_axis[i]),
            "put_moneyness": put_moneyness,
            "call_moneyness": call_moneyness,
        }

    def get_term_structure(self, target_moneyness: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        IV term structure at a target moneyness (IV vs time).

        Args:
            target_moneyness: Moneyness level (e.g. 1.0 for ATM).

        Returns:
            (times, IVs) 1D arrays of same length.
        """
        j = np.argmin(np.abs(self._M_axis - target_moneyness))
        times = self._T_axis.copy()
        ivs = self.IV_grid[:, j].copy()
        return times, ivs

    def check_calendar_arbitrage(
        self,
        target_moneyness: float = 1.0,
        tolerance: float = 0.01,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Check calendar arbitrage: IV*sqrt(T) should be non-decreasing in T.

        Args:
            target_moneyness: Moneyness level.
            tolerance: Allowed decrease (e.g. 0.01).

        Returns:
            (is_valid, violations_list). violations_list entries have T, IV, value, prev_value, index.
        """
        times, ivs = self.get_term_structure(target_moneyness)
        valid = np.isfinite(times) & np.isfinite(ivs)
        times = times[valid]
        ivs = ivs[valid]
        if len(times) < 2:
            return True, []
        # Sort by time to ensure ascending order for proper arbitrage check
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        ivs = ivs[sort_idx]
        values = ivs * np.sqrt(times)
        violations = []
        for i in range(1, len(values)):
            if values[i] < values[i - 1] - tolerance:
                violations.append({
                    "index": i,
                    "T": float(times[i]),
                    "IV": float(ivs[i]),
                    "value": float(values[i]),
                    "prev_value": float(values[i - 1]),
                })
        return (len(violations) == 0, violations)

    def generate_metrics_report(
        self,
        output_path: Optional[Union[str, bytes]] = None,
        target_times_years: Optional[List[float]] = None,
        target_moneyness: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Skew at several expiries, term structure, and arbitrage check.

        Args:
            output_path: If set, export metrics to CSV.
            target_times_years: Times (years) for skew (default [30/365, 90/365, 180/365]).
            target_moneyness: Moneyness for term structure and arbitrage check.

        Returns:
            Dict with skews, term_structure, calendar_arbitrage, summary.
        """
        if target_times_years is None:
            target_times_years = [30 / 365, 90 / 365, 180 / 365]

        skews = []
        for t in target_times_years:
            s = self.calculate_skew(t)
            skews.append(s)

        times, ivs = self.get_term_structure(target_moneyness)
        term_structure = {
            "times": times.tolist(),
            "ivs": ivs.tolist(),
            "target_moneyness": target_moneyness,
        }

        is_valid, violations = self.check_calendar_arbitrage(
            target_moneyness=target_moneyness, tolerance=0.01
        )
        calendar_arbitrage = {"is_valid": is_valid, "violations": violations}

        report = {
            "skews": skews,
            "term_structure": term_structure,
            "calendar_arbitrage": calendar_arbitrage,
            "summary": get_surface_summary(self.T_grid, self.M_grid, self.IV_grid),
        }

        if output_path:
            rows = []
            for s in skews:
                rows.append({
                    "metric": "skew",
                    "target_time": s["target_time"],
                    "T_used": s["T_used"],
                    "skew": s["skew"],
                    "iv_put": s["iv_put"],
                    "iv_call": s["iv_call"],
                })
            rows.append({
                "metric": "calendar_arbitrage",
                "target_moneyness": target_moneyness,
                "is_valid": is_valid,
                "n_violations": len(violations),
            })
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

        return report


def compare_surfaces(
    T_grid_1: np.ndarray,
    M_grid_1: np.ndarray,
    IV_grid_1: np.ndarray,
    T_grid_2: np.ndarray,
    M_grid_2: np.ndarray,
    IV_grid_2: np.ndarray,
    label1: str = "surface1",
    label2: str = "surface2",
) -> Dict[str, Any]:
    """
    Compare two surfaces (same grid shape required). Difference = IV_1 - IV_2.

    Args:
        T_grid_1, M_grid_1, IV_grid_1: First surface grid.
        T_grid_2, M_grid_2, IV_grid_2: Second surface grid.
        label1, label2: Labels for the report.

    Returns:
        Dict with diff_grid, mean_diff, max_diff, min_diff, label1, label2.
        diff_grid is IV_1 - IV_2 (NaN where either is non-finite).
    """
    if IV_grid_1.shape != IV_grid_2.shape:
        raise ValueError(
            f"Surfaces must have same shape; got {IV_grid_1.shape} and {IV_grid_2.shape}"
        )
    diff = np.where(
        np.isfinite(IV_grid_1) & np.isfinite(IV_grid_2),
        IV_grid_1 - IV_grid_2,
        np.nan,
    )
    valid = np.isfinite(diff)
    if not np.any(valid):
        return {
            "diff_grid": diff,
            "mean_diff": np.nan,
            "max_diff": np.nan,
            "min_diff": np.nan,
            "n_valid": 0,
            "label1": label1,
            "label2": label2,
        }
    return {
        "diff_grid": diff,
        "mean_diff": float(np.nanmean(diff)),
        "max_diff": float(np.nanmax(diff)),
        "min_diff": float(np.nanmin(diff)),
        "n_valid": int(valid.sum()),
        "label1": label1,
        "label2": label2,
    }
