"""
Module 6: SurfaceAnalytics

Extracts quantitative metrics from the constructed surface grid.
"""

from typing import Dict, Any

import numpy as np


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
