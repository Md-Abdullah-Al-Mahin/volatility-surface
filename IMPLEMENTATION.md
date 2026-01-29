# Implementation Plan: 3D Financial Volatility Surface Constructor

## Overview
This document describes the implementation of the volatility surface constructor: a four-module pipeline (data fetcher, IV processor, coordinate engine, interpolator) and a **live Streamlit dashboard** that runs the pipeline and displays an interactive 3D volatility surface that updates on a timer. The dashboard is the main entry point; there is no separate Visualization Suite or report-generation flow.

---

## Phase 1: Project Setup & Foundation (Week 1-2)

### Step 1.1: Environment Setup
- [ ] Create Python virtual environment (Python 3.9+)
- [ ] Initialize project structure:
  ```
  volatility-surface/
  ├── dashboard.py            # Streamlit live dashboard (entry point)
  ├── src/
  │   ├── __init__.py
  │   ├── data_fetcher.py      # Module 1
  │   ├── iv_processor.py      # Module 2
  │   ├── coordinate_engine.py  # Module 3
  │   ├── surface_interpolator.py # Module 4
  │   └── analytics.py         # Module 6 (placeholder)
  ├── tests/
  │   ├── __init__.py
  │   ├── test_data_fetcher.py
  │   ├── test_iv_processor.py
  │   ├── test_coordinate_engine.py
  │   └── test_interpolator.py
  ├── data/
  │   └── cache/              # For cached yfinance data
  ├── requirements.txt
  ├── setup.py
  └── README.md
  ```

- [ ] Create `requirements.txt` with dependencies:
  ```
  yfinance>=0.2.0
  pandas>=1.5.0
  numpy>=1.23.0
  scipy>=1.10.0
  plotly>=5.14.0
  pyarrow>=10.0.0
  streamlit>=1.28.0
  streamlit-autorefresh>=0.0.1
  pytest>=7.2.0
  ```

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Initialize git repository and create `.gitignore`
- [ ] Set up basic logging configuration

### Step 1.2: Module 1 - SmartOptionsDataFetcher

**File**: `src/data_fetcher.py`

**Implementation Steps**:
1. [ ] Create `SmartOptionsDataFetcher` class
2. [ ] Implement `__init__` method with optional cache directory parameter
3. [ ] Implement `fetch_options_data(ticker, date=None)` method:
   - Use `yfinance.Ticker(ticker)` to get ticker object
   - Get expiration dates: `ticker.options`
   - For each expiration:
     - Check cache first (if date matches)
     - Fetch calls: `ticker.option_chain(exp_date).calls`
     - Fetch puts: `ticker.option_chain(exp_date).puts`
     - Combine calls and puts into single DataFrame
     - Add expiration date column
     - Cache to disk (pickle or parquet format)
   - Return dictionary: `{exp_date: DataFrame, ...}`
4. [ ] Implement request throttling (time.sleep between requests)
5. [ ] Add error handling for:
   - Invalid ticker symbols
   - Network failures
   - Missing expiration dates
6. [ ] Implement cache management:
   - `_load_from_cache(ticker, exp_date)`
   - `_save_to_cache(ticker, exp_date, data)`
   - Cache invalidation logic (e.g., daily): when cache is invalid (stale or corrupt), delete the cache file
   - `clear_cache(ticker=None)` to explicitly invalidate and delete cache (per ticker or all)
7. [ ] Add logging for fetch operations
8. [ ] Write unit tests:
   - Test successful data fetch
   - Test caching mechanism
   - Test error handling
   - Test with multiple tickers (SPY, AAPL)

**Success Criteria**: Can fetch and cache SPY options data for all expirations, returning structured DataFrames.

### Step 1.3: Module 2 - ImpliedVolatilityProcessor

**File**: `src/iv_processor.py`

**Implementation Steps**:
1. [x] Create `ImpliedVolatilityProcessor` class
2. [x] Implement `__init__` method with risk-free rate source parameter
3. [x] Implement `_get_risk_free_rate()` method:
   - Fetch `^TNX` (10-year Treasury) from yfinance
   - Convert yield to decimal (divide by 100)
   - Fallback to user-provided constant (default: 0.05)
4. [x] Implement `_calculate_consensus_price(row)` method:
   - Priority: `(bid + ask) / 2` if both available
   - Fallback: `lastPrice` if available
   - Return NaN if neither available
5. [x] Implement Black-Scholes pricing functions:
   - `_black_scholes_price(S, K, T, r, sigma, option_type='call')`
   - Use `scipy.stats.norm` for cumulative normal distribution
   - Handle both call and put options
6. [x] Implement IV solver using Newton-Raphson method:
   - `_calculate_iv_newton_raphson(S, K, T, r, market_price, option_type, max_iter=100, tol=1e-6)`
   - Alternative: Implement Bisection method as fallback
   - Handle convergence failures gracefully
7. [x] Implement `_validate_iv(iv_value)` method:
   - Check for negative values → flag as invalid
   - Check for values > 2.0 (200%) → flag as invalid
   - Return boolean
8. [x] Implement main `process_iv(dataframes_dict, spot_price)` method:
   - Get risk-free rate
   - For each DataFrame in dictionary:
     - Calculate consensus price for each row
     - For each row:
       - If IV exists and is valid → keep it
       - Else → calculate IV using solver
     - Add/update `impliedVolatility` column
     - Log statistics (how many IVs were calculated vs. used from yfinance)
   - Return enhanced dictionary of DataFrames
9. [x] Add comprehensive error handling and logging
10. [x] Write unit tests:
    - Test Black-Scholes pricing (known values)
    - Test IV calculation (reverse: price → IV)
    - Test validation logic
    - Test with real data (SPY)
    - Test edge cases (very low/high strikes, very short/long expiry)

**Success Criteria**: Can process SPY data, validate IVs, and calculate missing IVs with accuracy within 0.5% of known values.

---

## Phase 2: Core Logic (Week 3-4)

### Step 2.1: Module 3 - SurfaceCoordinateEngine

**File**: `src/coordinate_engine.py`

**Implementation Steps**:
1. [x] Create `SurfaceCoordinateEngine` class
2. [x] Implement `__init__` method with moneyness method parameter ('ratio' or 'log')
3. [x] Implement `_calculate_time_to_expiry(exp_date, current_date)`:
   - Convert to annualized fraction: `(exp_date - current_date).days / 365.0`
   - Handle edge cases (same day, past dates)
4. [x] Implement `_calculate_moneyness(strike, spot, method='ratio')`:
   - If method == 'ratio': `M = strike / spot`
   - If method == 'log': `M = log(strike / spot)`
   - Return moneyness value
5. [x] Implement `_filter_extreme_moneyness(df, min_m=0.7, max_m=1.3)`:
   - Filter rows where moneyness is outside bounds
   - Log number of filtered rows
6. [x] Implement main `transform_to_coordinates(dataframes_dict, spot_price, moneyness_method='ratio', filter_extremes=True)`:
   - Initialize empty lists: `T`, `M`, `IV`
   - For each DataFrame in dictionary:
     - Extract expiration date
     - Calculate time to expiry
     - For each row:
       - Calculate moneyness
       - Extract IV (skip if NaN)
       - Append to T, M, IV lists
   - Apply filtering if enabled
   - Convert to numpy arrays
   - Return `(T, M, IV)` tuple
7. [x] Add data validation:
   - Check for empty arrays
   - Check for sufficient data points
   - Raise informative errors
8. [x] Write unit tests:
   - Test time calculation
   - Test moneyness calculation (both methods)
   - Test filtering logic
   - Test with real data
   - Test edge cases

**Success Criteria**: Can transform SPY options data into (T, M, IV) coordinate arrays with proper filtering.

### Step 2.2: Module 4 - VolatilitySurfaceInterpolator

**File**: `src/surface_interpolator.py`

**Implementation Steps**:
1. [x] Create `VolatilitySurfaceInterpolator` class
2. [x] Implement `__init__` method with interpolation method parameter ('cubic' or 'linear')
3. [x] Implement `_create_grid(T, M, n_points_T=50, n_points_M=50)`:
   - Calculate min/max for T and M
   - Create uniform grids using `numpy.linspace`
   - Return `T_grid, M_grid` as meshgrids
4. [x] Implement `interpolate_surface(T, M, IV, method='cubic', n_points_T=50, n_points_M=50)`:
   - Create grid
   - Use `scipy.interpolate.griddata`:
     - For cubic: `method='cubic'`
     - For linear: `method='linear'`
   - Handle extrapolation:
     - Use `fill_value='nearest'` or `fill_value=np.nan`
     - Optionally clip extrapolated values to reasonable bounds
   - Return `(T_grid, M_grid, IV_grid)`
5. [x] Add validation:
   - Check minimum number of data points (e.g., > 10)
   - Check for sufficient data coverage
   - Warn if extrapolation is extensive
6. [x] Implement optional smoothing (if needed):
   - Use `scipy.ndimage.gaussian_filter` for post-processing
7. [x] Write unit tests:
   - Test with synthetic data (known surface)
   - Test interpolation accuracy
   - Test edge cases (sparse data, boundary conditions)
   - Test both interpolation methods

**Success Criteria**: Can create smooth 50x50 grid of IV values from irregular (T, M, IV) points.

---

## Phase 3: Live Dashboard

### Step 3.1: Dashboard (dashboard.py)

**File**: `dashboard.py` (project root)

The **live volatility surface dashboard** is the main entry point. It runs the full pipeline and displays an interactive 3D Plotly surface that updates on a timer.

**Implementation (done)**:

1. **Run**: `streamlit run dashboard.py` — opens in browser (e.g. http://localhost:8501).

2. **Sidebar**:
   - **Ticker** — Underlying symbol (default `SPY`). User can change and refresh.
   - **Refresh every (seconds)** — Auto-refresh interval (e.g. 30–600 s). The app reruns and refetches when this interval has elapsed.
   - **Refresh now** — Button to manually trigger a refetch and redraw.

3. **Auto-refresh**:
   - Uses `streamlit-autorefresh` so the script reruns every N seconds (N = sidebar value).
   - On each run: if the refresh interval has passed, or ticker changed, or **Refresh now** was clicked, the pipeline is run again and the surface is updated.
   - `st.session_state` stores: `surface` (T_grid, M_grid, IV_grid), `surface_ticker`, `surface_spot`, `last_fetch_time`, `surface_error`.

4. **Pipeline in dashboard**:
   - `get_spot_price(ticker)` — Fetches current underlying price from yfinance (fast_info or history).
   - `build_surface(ticker)` — Runs: `SmartOptionsDataFetcher.fetch_options_data` → `ImpliedVolatilityProcessor.process_iv` → `SurfaceCoordinateEngine.transform_to_coordinates` → `VolatilitySurfaceInterpolator.interpolate_surface`; returns `(T_grid, M_grid, IV_grid, spot)`.

5. **Main area**:
   - **Metrics** — Ticker, spot price, grid size (e.g. 50×50).
   - **Plotly 3D surface** — Built inline with `plotly.graph_objects.Surface` (T_grid, M_grid, IV_grid). Interactive: rotate, zoom, pan in the browser. Displayed with `st.plotly_chart(fig, use_container_width=True)`.
   - **Caption** — Shows refresh interval and last load info.

6. **Error handling**: If fetch or pipeline fails, error message is stored in session state and shown; user can change ticker or click Refresh now.

**Success Criteria**: User runs `streamlit run dashboard.py`, sees a live 3D volatility surface that updates automatically at the chosen interval and when Refresh now is clicked.

---

## Phase 4: Analytics & Integration (Week 6)

### Step 4.1: Module 6 - SurfaceAnalytics

**File**: `src/analytics.py`

**Implementation Steps**:
1. [x] Create `SurfaceAnalytics` class
2. [x] Implement `__init__` method with surface grid data
3. [x] Implement `calculate_skew(target_time, put_moneyness=0.95, call_moneyness=1.05)`:
   - Find closest time point to target_time
   - Find IV at put_moneyness and call_moneyness
   - Calculate skew: `IV_put - IV_call`
   - Return skew value and metadata
4. [x] Implement `get_term_structure(target_moneyness=1.0)`:
   - Find closest moneyness point
   - Extract IV array across all times
   - Return (times, IVs) tuple
5. [x] Implement `check_calendar_arbitrage(target_moneyness, tolerance=0.01)`:
   - Extract IV term structure for target_moneyness
   - Calculate `IV * sqrt(T)` for each time point
   - Check if values are non-decreasing (with tolerance)
   - Flag violations
   - Return (is_valid, violations_list) tuple
6. [x] Implement `compare_surfaces(...)` (standalone function):
   - Calculate difference: `IV_grid_1 - IV_grid_2` (same grid shape required)
   - Return comparison report (diff_grid, mean_diff, max_diff, min_diff, labels)
7. [x] Implement `generate_metrics_report(output_path=None)`:
   - Calculate skew for multiple expiries (default 30/90/180 days)
   - Get term structure
   - Run arbitrage checks
   - Format as dictionary; export to CSV if path provided
   - Return report
8. [x] Write unit tests:
   - Test skew calculation
   - Test term structure extraction
   - Test arbitrage detection
   - Test surface comparison

**Success Criteria**: Can calculate all metrics and generate comprehensive analytics report for SPY surface.

**Dashboard integration**: The dashboard expander "Surface summary & analytics" uses `SurfaceAnalytics` to show skew at 30/90/180 days and calendar arbitrage (ATM) result.

### Step 4.2: Dashboard configurable options (no CLI)

**File**: `dashboard.py`

All pipeline and analytics options are configurable in the dashboard sidebar and expander. No CLI arguments.

**Implementation (done)**:

1. **Sidebar – main**
   - **Ticker** — Underlying symbol (default SPY).
   - **Refresh every (seconds)** — Auto-refresh interval (30–600 s).
   - **Refresh now** — Manually refetch and rebuild surface.
   - **Clear cache** — Clear cache for current ticker; next refresh fetches fresh data.

2. **Sidebar – expander "Pipeline & analytics options"**
   - **Risk-free rate (decimal)** — e.g. 0.05 = 5% (default 0.05).
   - **Use Treasury (^TNX) for risk-free rate** — Checkbox; if set, fetch 10Y Treasury yield (fallback to rate above).
   - **Moneyness** — `ratio` (strike/spot) or `log` (log(strike/spot)).
   - **Filter extreme moneyness** — Checkbox (default on).
   - **Min moneyness** / **Max moneyness** — Bounds when filtering (default 0.7, 1.3).
   - **Interpolation** — `cubic` or `linear`.
   - **Cache valid (hours)** — How long cache is considered fresh (default 24).

3. **Refetch when options change**
   - Current options are compared to the options used for the last surface (via a normalized key). If they differ, the next run refetches and rebuilds the surface.

4. **Export metrics**
   - In the "Surface summary & analytics" expander: **Export metrics to CSV** button. Downloads a CSV with skew (30/90/180 days) and calendar arbitrage result for the current surface.

5. **main.py**
   - Still only launches the dashboard (`streamlit run dashboard.py`). No argparse or CLI flags.

**Success Criteria**: All behaviour that would have been CLI flags (ticker, cache, risk-free rate, moneyness, interpolation, filter, export) is configurable and/or available on the dashboard.

---

## Phase 5: Testing & Refinement (Week 7)

### Step 5.1: Comprehensive Testing

**Implementation Steps**:
1. [ ] Write unit tests for each module (already started in previous phases)
2. [ ] Create integration tests:
   - Test full pipeline end-to-end
   - Test with multiple tickers (SPY, AAPL, QQQ)
   - Test error recovery
3. [ ] Create performance tests:
   - Measure execution time (target: < 15 seconds)
   - Profile bottlenecks
   - Optimize if needed
4. [ ] Create robustness tests:
   - Test with missing data scenarios
   - Test with extreme market conditions
   - Test with various ticker types
5. [ ] Set up continuous testing:
   - Configure pytest
   - Add coverage reporting
   - Target: > 80% code coverage
6. [ ] Fix all identified bugs and edge cases

### Step 5.2: Code Quality & Documentation

**Implementation Steps**:
1. [ ] Add comprehensive docstrings to all classes and methods:
   - Google or NumPy style docstrings
   - Include parameters, returns, examples
2. [ ] Add type hints throughout codebase
3. [ ] Run code formatter (black, autopep8)
4. [ ] Run linter (pylint, flake8)
5. [ ] Fix all warnings and style issues
6. [ ] Update README.md with:
   - Installation instructions
   - Usage examples
   - Module overview
   - API documentation links
7. [ ] Create `CONTRIBUTING.md` if needed

---

## Phase 6: Stretch Goals (Week 8)

### Step 6.1: Advanced Features (Optional)

**Implementation Steps**:
1. [ ] Research and implement SVI (Stochastic Volatility Inspired) model:
   - More sophisticated surface parameterization
   - Better handling of volatility smile
2. [x] Build Streamlit dashboard (done): interactive ticker selection, real-time surface updates via auto-refresh.
3. [ ] Add yield curve support:
   - Fetch multiple Treasury rates
   - Use maturity-matched risk-free rates
4. [ ] Implement surface calibration:
   - Fit parametric models to surface
   - Compare model vs. market data

---

## Implementation Checklist Summary

### Week 1-2: Foundation
- [ ] Project setup and environment
- [ ] Module 1: Data Fetcher (complete)
- [ ] Module 2: IV Processor (complete)
- [ ] Basic tests for Modules 1 & 2

### Week 3-4: Core Logic
- [ ] Module 3: Coordinate Engine (complete)
- [ ] Module 4: Surface Interpolator (complete)
- [ ] Tests for Modules 3 & 4

### Week 5: Dashboard
- [x] Live dashboard (dashboard.py)
- [x] Streamlit + streamlit-autorefresh
- [x] Interactive 3D Plotly surface with auto-refresh

### Week 6: Analytics & Integration
- [ ] Module 6: Analytics (complete)
- [ ] CLI implementation (main.py)
- [ ] Integration tests

### Week 7: Testing & Polish
- [ ] Comprehensive test suite
- [ ] Code documentation
- [ ] Performance optimization
- [ ] README updates

### Week 8: Stretch Goals
- [ ] Advanced features (optional)

---

## Key Implementation Notes

1. **Incremental Development**: Build and test each module independently before integration.
2. **Data Validation**: Always validate inputs and handle edge cases gracefully.
3. **Logging**: Use Python's `logging` module throughout for debugging and monitoring.
4. **Error Handling**: Implement try-except blocks with informative error messages.
5. **Performance**: Profile code early and optimize bottlenecks (especially IV calculation).
6. **Testing**: Write tests as you develop, not after. Use fixtures for common test data.
7. **Documentation**: Keep docstrings up-to-date as you code.
8. **Version Control**: Commit frequently with descriptive messages.

---

## Success Metrics

Track these KPIs throughout implementation:
- **Accuracy**: IV calculations within 0.5% of known values
- **Performance**: Full pipeline < 15 seconds
- **Robustness**: 95% success rate on trading days
- **Code Quality**: > 80% test coverage, no critical linter errors
- **Usability**: CLI commands work with < 5 arguments

---

## Next Steps

1. Start with Phase 1, Step 1.1 (Environment Setup)
2. Complete each step before moving to the next
3. Test thoroughly at each milestone
4. Document decisions and challenges as you go
5. Iterate based on test results and performance profiling

Good luck with the implementation!
