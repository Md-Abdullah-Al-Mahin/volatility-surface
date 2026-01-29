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

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from src.data_fetcher import SmartOptionsDataFetcher
from src.iv_processor import ImpliedVolatilityProcessor
from src.coordinate_engine import SurfaceCoordinateEngine
from src.surface_interpolator import VolatilitySurfaceInterpolator
from src.analytics import get_surface_summary, SurfaceAnalytics

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


def build_surface(ticker: str, options: dict):
    """Fetch options, process IV, build (T_grid, M_grid, IV_grid). Uses options from dashboard."""
    spot = get_spot_price(ticker)
    if spot <= 0:
        raise ValueError(f"Could not get spot price for {ticker}")
    fetcher = SmartOptionsDataFetcher(cache_valid_hours=options.get("cache_valid_hours", 24))
    data = fetcher.fetch_options_data(ticker)
    iv_processor = ImpliedVolatilityProcessor(
        risk_free_rate=options.get("risk_free_rate", 0.05),
        use_treasury_rate=options.get("use_treasury_rate", False),
    )
    data = iv_processor.process_iv(data, spot_price=spot)
    engine = SurfaceCoordinateEngine(moneyness_method=options.get("moneyness_method", "ratio"))
    T, M, IV = engine.transform_to_coordinates(
        data,
        spot_price=spot,
        filter_extremes=options.get("filter_extremes", True),
        min_m=options.get("min_m", 0.7),
        max_m=options.get("max_m", 1.3),
    )
    interp = VolatilitySurfaceInterpolator(method=options.get("interp_method", "cubic"))
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

        st.write("---")
        with st.expander("Pipeline & analytics options"):
            risk_free_rate = st.number_input(
                "Risk-free rate (decimal)",
                min_value=0.0,
                max_value=0.2,
                value=0.05,
                step=0.005,
                format="%.3f",
                help="e.g. 0.05 = 5%",
            )
            use_treasury_rate = st.checkbox(
                "Use Treasury (^TNX) for risk-free rate",
                value=False,
                help="Fetch 10Y Treasury yield; fallback to rate above",
            )
            moneyness_method = st.selectbox(
                "Moneyness",
                options=["ratio", "log"],
                index=0,
                help="ratio = strike/spot, log = log(strike/spot)",
            )
            filter_extremes = st.checkbox("Filter extreme moneyness", value=True)
            min_m = st.number_input("Min moneyness", min_value=0.5, max_value=1.0, value=0.7, step=0.05)
            max_m = st.number_input("Max moneyness", min_value=1.0, max_value=2.0, value=1.3, step=0.05)
            interp_method = st.selectbox(
                "Interpolation",
                options=["cubic", "linear"],
                index=0,
            )
            cache_valid_hours = st.number_input(
                "Cache valid (hours)",
                min_value=1,
                max_value=168,
                value=24,
                step=1,
            )

        options = {
            "risk_free_rate": risk_free_rate,
            "use_treasury_rate": use_treasury_rate,
            "moneyness_method": moneyness_method,
            "filter_extremes": filter_extremes,
            "min_m": min_m,
            "max_m": max_m,
            "interp_method": interp_method,
            "cache_valid_hours": cache_valid_hours,
        }

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
    if "surface_options" not in st.session_state:
        st.session_state.surface_options = None
    if "surface_options_key" not in st.session_state:
        st.session_state.surface_options_key = None

    def _options_key(op):
        return tuple(sorted(
            (k, round(v, 8) if isinstance(v, float) else v)
            for k, v in op.items()
        ))
    options_key = _options_key(options)
    options_changed = (
        st.session_state.surface_options_key is not None
        and st.session_state.surface_options_key != options_key
    )
    now = time.time()
    elapsed = (now - st.session_state.last_fetch_time) if st.session_state.last_fetch_time else float("inf")
    need_fetch = (
        refresh_now
        or st.session_state.surface is None
        or st.session_state.surface_ticker != ticker
        or options_changed
        or elapsed >= refresh_seconds
    )

    if need_fetch:
        with st.spinner(f"Loading {ticker} options and building surface…"):
            try:
                T_grid, M_grid, IV_grid, spot = build_surface(ticker, options)
                st.session_state.surface = (T_grid, M_grid, IV_grid)
                st.session_state.surface_ticker = ticker
                st.session_state.surface_spot = spot
                st.session_state.surface_options = options.copy()
                st.session_state.surface_options_key = options_key
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

    # Surface summary and analytics
    summary = get_surface_summary(T_grid, M_grid, IV_grid)
    analytics = SurfaceAnalytics(T_grid, M_grid, IV_grid)
    with st.expander("Surface summary & analytics"):
        if summary["n_valid"]:
            st.write(f"**IV range:** {summary['min_iv']:.2%} – {summary['max_iv']:.2%}")
            st.write(f"**Mean IV:** {summary['mean_iv']:.2%}")
            st.write(f"**Valid points:** {summary['n_valid']}")
        st.write(f"**Time to expiry (years):** {summary['T_range'][0]:.3f} – {summary['T_range'][1]:.3f}")
        st.write(f"**Moneyness:** {summary['M_range'][0]:.3f} – {summary['M_range'][1]:.3f}")
        st.write("---")
        st.write("**Skew (IV_put − IV_call) at 30 / 90 / 180 days:**")
        for days in [30, 90, 180]:
            t_years = days / 365.0
            s = analytics.calculate_skew(t_years, put_moneyness=0.95, call_moneyness=1.05)
            skew_val = s["skew"]
            st.write(f"  {days}d: skew = {skew_val:.4f}" if np.isfinite(skew_val) else f"  {days}d: —")
        is_valid, violations = analytics.check_calendar_arbitrage(target_moneyness=1.0, tolerance=0.01)
        st.write(f"**Calendar arbitrage (ATM):** {'✓ No violations' if is_valid else f'⚠ {len(violations)} violation(s)'}")
        st.write("---")
        report = analytics.generate_metrics_report(target_moneyness=1.0)
        rows = []
        for s in report["skews"]:
            rows.append({
                "metric": "skew",
                "target_time_years": s["target_time"],
                "T_used": s["T_used"],
                "skew": s["skew"],
                "iv_put": s["iv_put"],
                "iv_call": s["iv_call"],
            })
        rows.append({
            "metric": "calendar_arbitrage",
            "target_moneyness": 1.0,
            "is_valid": report["calendar_arbitrage"]["is_valid"],
            "n_violations": len(report["calendar_arbitrage"]["violations"]),
        })
        csv_df = pd.DataFrame(rows)
        csv_str = csv_df.to_csv(index=False)
        st.download_button(
            "Export metrics to CSV",
            data=csv_str,
            file_name=f"{ticker_show}_metrics.csv",
            mime="text/csv",
            key="export_metrics",
        )

    # 3D Plotly surface (interactive)
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
