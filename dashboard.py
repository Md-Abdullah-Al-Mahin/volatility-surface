"""
Live Volatility Surface Dashboard

Streamlit app that shows the 3D volatility surface and updates on a timer.
Run: streamlit run dashboard.py
"""

import sys
import time
from pathlib import Path

# Ensure src is on path when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import yfinance as yf

from src.data_fetcher import SmartOptionsDataFetcher
from src.iv_processor import ImpliedVolatilityProcessor
from src.coordinate_engine import SurfaceCoordinateEngine
from src.surface_interpolator import VolatilitySurfaceInterpolator
from src.analytics import get_surface_summary

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


def get_spot_price(ticker: str) -> float:
    """Fetch current underlying price from yfinance."""
    try:
        t = yf.Ticker(ticker)
        if hasattr(t, "fast_info") and getattr(t.fast_info, "last_price", None) is not None:
            return float(t.fast_info.last_price)
        hist = t.history(period="5d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def build_surface(ticker: str):
    """Fetch options, process IV, build (T_grid, M_grid, IV_grid). Fails fast on invalid ticker (spot first)."""
    spot = get_spot_price(ticker)
    if spot <= 0:
        raise ValueError(f"Could not get spot price for {ticker}")
    fetcher = SmartOptionsDataFetcher(cache_valid_hours=24)
    data = fetcher.fetch_options_data(ticker)
    iv_processor = ImpliedVolatilityProcessor(risk_free_rate=0.05, use_treasury_rate=False)
    data = iv_processor.process_iv(data, spot_price=spot)
    engine = SurfaceCoordinateEngine(moneyness_method="ratio")
    T, M, IV = engine.transform_to_coordinates(data, spot_price=spot, filter_extremes=True)
    interp = VolatilitySurfaceInterpolator(method="cubic")
    T_grid, M_grid, IV_grid = interp.interpolate_surface(T, M, IV)
    return T_grid, M_grid, IV_grid, spot


def main():
    st.set_page_config(
        page_title="Volatility Surface",
        page_icon="📈",
        layout="wide",
    )
    st.title("Live Volatility Surface")
    st.caption("3D surface updates automatically. Rotate, zoom, and pan in the plot.")

    # Sidebar
    with st.sidebar:
        ticker = st.text_input("Ticker", value="SPY", max_chars=10).strip().upper() or "SPY"
        refresh_seconds = st.number_input(
            "Refresh every (seconds)",
            min_value=30,
            max_value=600,
            value=120,
            step=30,
        )
        refresh_now = st.button("Refresh now")
        clear_cache = st.button("Clear cache")
        if clear_cache:
            fetcher = SmartOptionsDataFetcher()
            n = fetcher.clear_cache(ticker)
            st.session_state.surface = None
            st.session_state.last_fetch_time = None
            st.session_state.surface_error = None
            st.success(f"Cleared {n} cache file(s). Surface will refetch on next refresh.")
            st.rerun()

    # Auto-refresh (rerun script every N seconds)
    if st_autorefresh is not None:
        st_autorefresh(interval=refresh_seconds * 1000, key="vol_refresh")
    else:
        st.sidebar.warning("Install streamlit-autorefresh for auto-updates: pip install streamlit-autorefresh")

    # Session state for cached surface
    if "surface" not in st.session_state:
        st.session_state.surface = None
    if "surface_ticker" not in st.session_state:
        st.session_state.surface_ticker = None
    if "surface_spot" not in st.session_state:
        st.session_state.surface_spot = None
    if "surface_error" not in st.session_state:
        st.session_state.surface_error = None
    if "last_fetch_time" not in st.session_state:
        st.session_state.last_fetch_time = None

    # Refetch if: manual refresh, no data, ticker changed, or refresh_seconds elapsed
    now = time.time()
    elapsed = (now - st.session_state.last_fetch_time) if st.session_state.last_fetch_time else float("inf")
    need_fetch = (
        refresh_now
        or st.session_state.surface is None
        or st.session_state.surface_ticker != ticker
        or elapsed >= refresh_seconds
    )

    if need_fetch:
        with st.spinner(f"Loading {ticker} options and building surface…"):
            try:
                T_grid, M_grid, IV_grid, spot = build_surface(ticker)
                st.session_state.surface = (T_grid, M_grid, IV_grid)
                st.session_state.surface_ticker = ticker
                st.session_state.surface_spot = spot
                st.session_state.last_fetch_time = time.time()
                st.session_state.surface_error = None
            except Exception as e:
                st.session_state.surface_error = str(e)
                st.session_state.surface = None

    if st.session_state.surface_error:
        st.error(st.session_state.surface_error)
        st.info("Check ticker symbol and network, then click Refresh now.")
        return

    if st.session_state.surface is None:
        st.info("Click **Refresh now** or wait for auto-refresh to load the surface.")
        return

    T_grid, M_grid, IV_grid = st.session_state.surface
    spot = st.session_state.surface_spot
    ticker_show = st.session_state.surface_ticker

    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ticker", ticker_show)
    with col2:
        st.metric("Spot", f"${spot:.2f}")
    with col3:
        st.metric("Grid", f"{IV_grid.shape[0]}×{IV_grid.shape[1]}")

    # Surface summary (analytics)
    summary = get_surface_summary(T_grid, M_grid, IV_grid)
    with st.expander("Surface summary"):
        if summary["n_valid"]:
            st.write(f"**IV range:** {summary['min_iv']:.2%} – {summary['max_iv']:.2%}")
            st.write(f"**Mean IV:** {summary['mean_iv']:.2%}")
            st.write(f"**Valid points:** {summary['n_valid']}")
        st.write(f"**Time to expiry (years):** {summary['T_range'][0]:.3f} – {summary['T_range'][1]:.3f}")
        st.write(f"**Moneyness:** {summary['M_range'][0]:.3f} – {summary['M_range'][1]:.3f}")

    # 3D Plotly surface (interactive)
    import numpy as np
    import plotly.graph_objects as go

    IV_plot = np.where(np.isfinite(IV_grid), IV_grid, np.nan)
    fig = go.Figure(
        data=[
            go.Surface(
                x=T_grid,
                y=M_grid,
                z=IV_plot,
                colorscale="Viridis",
                colorbar=dict(title="IV"),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"{ticker_show} Volatility Surface", font=dict(size=18)),
        scene=dict(
            xaxis_title="Time to Expiry (years)",
            yaxis_title="Moneyness",
            zaxis_title="Implied Volatility",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=700,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Auto-refresh every {refresh_seconds}s. Last load: {ticker_show} @ ${spot:.2f}")


if __name__ == "__main__":
    main()
