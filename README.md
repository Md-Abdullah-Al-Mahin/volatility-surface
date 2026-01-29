# Volatility Surface – Live Dashboard

Live 3D volatility surface from options data. Fetches options via yfinance, processes IV, interpolates to a grid, and shows an interactive Plotly 3D surface that updates on a timer.

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Open the URL (e.g. http://localhost:8501). Use the sidebar to set the ticker (default SPY) and refresh interval. The 3D surface is interactive (rotate, zoom, pan) and refreshes automatically.

## Project layout

- **dashboard.py** – Streamlit app (entry point)
- **src/** – Pipeline: data_fetcher → iv_processor → coordinate_engine → surface_interpolator
- **data/cache/** – Cached options data
- **tests/** – Unit tests for the pipeline
