"""
Module 4: VolatilitySurfaceInterpolator

Fits a smooth, continuous surface to discrete, irregularly-spaced (T, M, IV) points
using scipy.interpolate.griddata (linear or cubic).
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

MIN_POINTS = 10
DEFAULT_GRID_SIZE_T = 50
DEFAULT_GRID_SIZE_M = 50
IV_MIN = 0.0
IV_MAX = 2.0


class VolatilitySurfaceInterpolator:
    """
    Interpolates scattered (T, M, IV) points onto a regular 2D grid
    for visualization and analytics.
    """

    def __init__(self, method: str = "cubic"):
        """
        Initialize the interpolator.

        Args:
            method: 'cubic' or 'linear' (passed to scipy.interpolate.griddata).
        """
        if method not in ("cubic", "linear"):
            raise ValueError(f"method must be 'cubic' or 'linear', got {method!r}")
        self.method = method
        logger.info(f"Initialized VolatilitySurfaceInterpolator with method={method!r}")

    def _create_grid(
        self,
        T: np.ndarray,
        M: np.ndarray,
        n_points_T: int = DEFAULT_GRID_SIZE_T,
        n_points_M: int = DEFAULT_GRID_SIZE_M,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build regular 1D grids and 2D meshgrids for T and M.

        Args:
            T, M: 1D arrays of observed coordinates.
            n_points_T: Number of grid points along T.
            n_points_M: Number of grid points along M.

        Returns:
            (T_grid, M_grid) 2D arrays of shape (n_points_T, n_points_M) with
            T_grid[i, j] = T_1d[i], M_grid[i, j] = M_1d[j].
        """
        t_min, t_max = float(np.min(T)), float(np.max(T))
        m_min, m_max = float(np.min(M)), float(np.max(M))
        # Slight padding to avoid boundary issues
        t_pad = max((t_max - t_min) * 0.01, 1e-6) if t_max > t_min else 1e-6
        m_pad = max((m_max - m_min) * 0.01, 1e-6) if m_max > m_min else 1e-6
        t_1d = np.linspace(t_min - t_pad, t_max + t_pad, n_points_T)
        m_1d = np.linspace(m_min - m_pad, m_max + m_pad, n_points_M)
        T_grid, M_grid = np.meshgrid(t_1d, m_1d, indexing="ij")
        return T_grid, M_grid

    def interpolate_surface(
        self,
        T: np.ndarray,
        M: np.ndarray,
        IV: np.ndarray,
        method: Optional[str] = None,
        n_points_T: int = DEFAULT_GRID_SIZE_T,
        n_points_M: int = DEFAULT_GRID_SIZE_M,
        fill_value: Optional[float] = np.nan,
        clip_iv: bool = True,
        sigma_smooth: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate scattered (T, M, IV) onto a regular grid.

        Args:
            T, M, IV: 1D arrays of same length (from SurfaceCoordinateEngine).
            method: Override instance method ('cubic' or 'linear').
            n_points_T: Grid size along time dimension.
            n_points_M: Grid size along moneyness dimension.
            fill_value: Value for points outside convex hull (default: np.nan).
            clip_iv: If True, clip IV grid to [IV_MIN, IV_MAX] after interpolation.
            sigma_smooth: If set, apply Gaussian smoothing (sigma in grid cells).

        Returns:
            (T_grid, M_grid, IV_grid) where IV_grid has shape (n_points_T, n_points_M).

        Raises:
            ValueError: If too few points or invalid inputs.
        """
        T = np.asarray(T, dtype=float)
        M = np.asarray(M, dtype=float)
        IV = np.asarray(IV, dtype=float)
        if T.size != M.size or M.size != IV.size:
            raise ValueError(
                f"T, M, IV must have same length; got {T.size}, {M.size}, {IV.size}"
            )
        n = len(T)
        if n < MIN_POINTS:
            raise ValueError(
                f"Need at least {MIN_POINTS} points for interpolation, got {n}"
            )
        # Validate that T values are non-negative (should be time to expiry in years)
        if np.any(T < 0):
            n_negative = np.sum(T < 0)
            logger.warning(
                f"Found {n_negative} negative T values (time to expiry). "
                "These will be filtered out or may cause issues."
            )
            # Filter out negative T values
            valid_mask = T >= 0
            if np.sum(valid_mask) < MIN_POINTS:
                raise ValueError(
                    f"After filtering negative T values, only {np.sum(valid_mask)} points remain, "
                    f"need at least {MIN_POINTS}"
                )
            T = T[valid_mask]
            M = M[valid_mask]
            IV = IV[valid_mask]
            n = len(T)
            logger.info(f"Filtered to {n} points with non-negative T")

        method = method or self.method
        points = np.column_stack((T, M))

        T_grid, M_grid = self._create_grid(T, M, n_points_T, n_points_M)
        
        # Create evaluation points as 2D coordinate pairs to ensure consistent 2D output
        # Flatten the meshgrids and stack them to create (T, M) pairs
        eval_points = np.column_stack((T_grid.ravel(), M_grid.ravel()))
        
        # Call griddata with the flattened evaluation points
        IV_grid_flat = griddata(
            points,
            IV,
            eval_points,
            method=method,
            fill_value=fill_value if fill_value is not None else np.nan,
        )
        if IV_grid_flat is None:
            raise ValueError("griddata returned None")
        
        # Reshape the flattened result to match the grid shape
        if IV_grid_flat.ndim != 1:
            raise ValueError(f"Expected 1D result from griddata with eval_points, got {IV_grid_flat.ndim}D")
        if IV_grid_flat.size != n_points_T * n_points_M:
            raise ValueError(f"Unexpected IV_grid size: {IV_grid_flat.size}, expected {n_points_T * n_points_M}")
        
        # Reshape to 2D grid matching T_grid and M_grid shape
        IV_grid = IV_grid_flat.reshape(n_points_T, n_points_M)

        # Count extrapolation (NaN outside convex hull)
        n_nan = np.isnan(IV_grid).sum()
        n_total = IV_grid.size
        if n_nan > 0.5 * n_total:
            logger.warning(
                f"More than half of grid points are extrapolated ({n_nan}/{n_total}). "
                "Consider expanding data range or using fill_value='nearest'."
            )

        if clip_iv:
            IV_grid = np.clip(IV_grid, IV_MIN, IV_MAX)

        if sigma_smooth is not None and sigma_smooth > 0:
            # Only smooth non-NaN; use 0 for NaN then mask back
            mask = np.isnan(IV_grid)
            filled = np.where(mask, 0.0, IV_grid)
            smoothed = ndimage.gaussian_filter(filled, sigma=sigma_smooth, mode="nearest")
            IV_grid = np.where(mask, np.nan, smoothed)

        logger.info(
            f"interpolate_surface: grid {IV_grid.shape}, "
            f"extrapolated cells: {n_nan}/{n_total}"
        )
        return T_grid, M_grid, IV_grid
