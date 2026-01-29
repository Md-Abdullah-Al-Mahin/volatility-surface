# Implementation Plan: 3D Financial Volatility Surface Constructor

## Overview
This document provides a step-by-step implementation guide for building the volatility surface constructor. Follow these steps sequentially, with each module building upon the previous ones.

---

## Phase 1: Project Setup & Foundation (Week 1-2)

### Step 1.1: Environment Setup
- [ ] Create Python virtual environment (Python 3.9+)
- [ ] Initialize project structure:
  ```
  volatility-surface/
  ├── src/
  │   ├── __init__.py
  │   ├── data_fetcher.py      # Module 1
  │   ├── iv_processor.py      # Module 2
  │   ├── coordinate_engine.py  # Module 3
  │   ├── surface_interpolator.py # Module 4
  │   ├── visualization.py     # Module 5
  │   └── analytics.py         # Module 6
  ├── tests/
  │   ├── __init__.py
  │   ├── test_data_fetcher.py
  │   ├── test_iv_processor.py
  │   ├── test_coordinate_engine.py
  │   ├── test_interpolator.py
  │   ├── test_visualization.py
  │   └── test_analytics.py
  ├── notebooks/
  │   └── demo.ipynb
  ├── data/
  │   └── cache/              # For cached yfinance data
  ├── main.py                 # CLI entry point
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
  matplotlib>=3.6.0
  plotly>=5.14.0
  pytest>=7.2.0
  jupyter>=1.0.0
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
1. [ ] Create `ImpliedVolatilityProcessor` class
2. [ ] Implement `__init__` method with risk-free rate source parameter
3. [ ] Implement `_get_risk_free_rate()` method:
   - Fetch `^TNX` (10-year Treasury) from yfinance
   - Convert yield to decimal (divide by 100)
   - Fallback to user-provided constant (default: 0.05)
4. [ ] Implement `_calculate_consensus_price(row)` method:
   - Priority: `(bid + ask) / 2` if both available
   - Fallback: `lastPrice` if available
   - Return NaN if neither available
5. [ ] Implement Black-Scholes pricing functions:
   - `_black_scholes_price(S, K, T, r, sigma, option_type='call')`
   - Use `scipy.stats.norm` for cumulative normal distribution
   - Handle both call and put options
6. [ ] Implement IV solver using Newton-Raphson method:
   - `_calculate_iv_newton_raphson(S, K, T, r, market_price, option_type, max_iter=100, tol=1e-6)`
   - Alternative: Implement Bisection method as fallback
   - Handle convergence failures gracefully
7. [ ] Implement `_validate_iv(iv_value)` method:
   - Check for negative values → flag as invalid
   - Check for values > 2.0 (200%) → flag as invalid
   - Return boolean
8. [ ] Implement main `process_iv(dataframes_dict, spot_price)` method:
   - Get risk-free rate
   - For each DataFrame in dictionary:
     - Calculate consensus price for each row
     - For each row:
       - If IV exists and is valid → keep it
       - Else → calculate IV using solver
     - Add/update `impliedVolatility` column
     - Log statistics (how many IVs were calculated vs. used from yfinance)
   - Return enhanced dictionary of DataFrames
9. [ ] Add comprehensive error handling and logging
10. [ ] Write unit tests:
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
1. [ ] Create `SurfaceCoordinateEngine` class
2. [ ] Implement `__init__` method with moneyness method parameter ('ratio' or 'log')
3. [ ] Implement `_calculate_time_to_expiry(exp_date, current_date)`:
   - Convert to annualized fraction: `(exp_date - current_date).days / 365.0`
   - Handle edge cases (same day, past dates)
4. [ ] Implement `_calculate_moneyness(strike, spot, method='ratio')`:
   - If method == 'ratio': `M = strike / spot`
   - If method == 'log': `M = log(strike / spot)`
   - Return moneyness value
5. [ ] Implement `_filter_extreme_moneyness(df, min_m=0.7, max_m=1.3)`:
   - Filter rows where moneyness is outside bounds
   - Log number of filtered rows
6. [ ] Implement main `transform_to_coordinates(dataframes_dict, spot_price, moneyness_method='ratio', filter_extremes=True)`:
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
7. [ ] Add data validation:
   - Check for empty arrays
   - Check for sufficient data points
   - Raise informative errors
8. [ ] Write unit tests:
   - Test time calculation
   - Test moneyness calculation (both methods)
   - Test filtering logic
   - Test with real data
   - Test edge cases

**Success Criteria**: Can transform SPY options data into (T, M, IV) coordinate arrays with proper filtering.

### Step 2.2: Module 4 - VolatilitySurfaceInterpolator

**File**: `src/surface_interpolator.py`

**Implementation Steps**:
1. [ ] Create `VolatilitySurfaceInterpolator` class
2. [ ] Implement `__init__` method with interpolation method parameter ('cubic' or 'linear')
3. [ ] Implement `_create_grid(T, M, n_points_T=50, n_points_M=50)`:
   - Calculate min/max for T and M
   - Create uniform grids using `numpy.linspace`
   - Return `T_grid, M_grid` as meshgrids
4. [ ] Implement `interpolate_surface(T, M, IV, method='cubic', n_points_T=50, n_points_M=50)`:
   - Create grid
   - Use `scipy.interpolate.griddata`:
     - For cubic: `method='cubic'`
     - For linear: `method='linear'`
   - Handle extrapolation:
     - Use `fill_value='nearest'` or `fill_value=np.nan`
     - Optionally clip extrapolated values to reasonable bounds
   - Return `(T_grid, M_grid, IV_grid)`
5. [ ] Add validation:
   - Check minimum number of data points (e.g., > 10)
   - Check for sufficient data coverage
   - Warn if extrapolation is extensive
6. [ ] Implement optional smoothing (if needed):
   - Use `scipy.ndimage.gaussian_filter` for post-processing
7. [ ] Write unit tests:
   - Test with synthetic data (known surface)
   - Test interpolation accuracy
   - Test edge cases (sparse data, boundary conditions)
   - Test both interpolation methods

**Success Criteria**: Can create smooth 50x50 grid of IV values from irregular (T, M, IV) points.

---

## Phase 3: Visualization (Week 5)

### Step 3.1: Module 5 - VisualizationSuite

**File**: `src/visualization.py`

**Implementation Steps**:
1. [ ] Create `VisualizationSuite` class
2. [ ] Implement `__init__` method with output directory parameter
3. [ ] Implement `plot_3d_surface_matplotlib(T_grid, M_grid, IV_grid, title, save_path=None)`:
   - Use `matplotlib.pyplot` and `mpl_toolkits.mplot3d`
   - Create `ax = fig.add_subplot(111, projection='3d')`
   - Use `ax.plot_surface(T_grid, M_grid, IV_grid, cmap='viridis')`
   - Set labels: X='Time to Expiry', Y='Moneyness', Z='Implied Volatility'
   - Add colorbar
   - Save if path provided
   - Return figure object
4. [ ] Implement `plot_3d_surface_plotly(T_grid, M_grid, IV_grid, title, save_path=None)`:
   - Use `plotly.graph_objects.Surface`
   - Create interactive plot with hover information
   - Add axis labels and title
   - Save as HTML if path provided
   - Return figure object
5. [ ] Implement `plot_volatility_smile(T_grid, M_grid, IV_grid, target_times=[30, 90, 180], save_path=None)`:
   - For each target time (in days):
     - Find closest time point in T_grid
     - Extract IV slice for that time (IV vs M)
     - Plot IV vs Moneyness
   - Add legend with expiry labels
   - Save if path provided
   - Return figure object
6. [ ] Implement `plot_term_structure(T_grid, M_grid, IV_grid, target_moneyness=1.0, save_path=None)`:
   - Find closest moneyness point to target (e.g., M=1.0)
   - Extract IV slice for that moneyness (IV vs T)
   - Plot IV vs Time to Expiry
   - Add labels and title
   - Save if path provided
   - Return figure object
7. [ ] Implement `generate_all_plots(T_grid, M_grid, IV_grid, ticker, output_dir)`:
   - Generate all four plot types
   - Save with descriptive filenames
   - Return dictionary of figure objects
8. [ ] Add styling and formatting:
   - Professional color schemes
   - Consistent font sizes
   - Grid lines where appropriate
9. [ ] Write unit tests:
   - Test plot generation (check files created)
   - Test with various data shapes
   - Test error handling

**Success Criteria**: Can generate all four visualization types for SPY data, producing publication-quality plots.

---

## Phase 4: Analytics & Integration (Week 6)

### Step 4.1: Module 6 - SurfaceAnalytics

**File**: `src/analytics.py`

**Implementation Steps**:
1. [ ] Create `SurfaceAnalytics` class
2. [ ] Implement `__init__` method with surface grid data
3. [ ] Implement `calculate_skew(target_time, put_moneyness=0.95, call_moneyness=1.05)`:
   - Find closest time point to target_time
   - Find IV at put_moneyness and call_moneyness
   - Calculate skew: `IV_put - IV_call`
   - Return skew value and metadata
4. [ ] Implement `get_term_structure(target_moneyness=1.0)`:
   - Find closest moneyness point
   - Extract IV array across all times
   - Return (times, IVs) tuple
5. [ ] Implement `check_calendar_arbitrage(target_moneyness, tolerance=0.01)`:
   - Extract IV term structure for target_moneyness
   - Calculate `IV * sqrt(T)` for each time point
   - Check if values are non-decreasing (with tolerance)
   - Flag violations
   - Return (is_valid, violations_list) tuple
6. [ ] Implement `compare_surfaces(surface1, surface2, label1, label2)`:
   - Calculate difference: `IV_grid_1 - IV_grid_2`
   - Generate difference plot
   - Calculate summary statistics (mean diff, max diff, etc.)
   - Return comparison report dictionary
7. [ ] Implement `generate_metrics_report(output_path=None)`:
   - Calculate skew for multiple expiries
   - Get term structure
   - Run arbitrage checks
   - Format as dictionary or DataFrame
   - Export to CSV if path provided
   - Return report
8. [ ] Write unit tests:
   - Test skew calculation
   - Test term structure extraction
   - Test arbitrage detection
   - Test surface comparison

**Success Criteria**: Can calculate all metrics and generate comprehensive analytics report for SPY surface.

### Step 4.2: CLI Integration - main.py

**File**: `main.py`

**Implementation Steps**:
1. [ ] Set up argument parser using `argparse`:
   - `--ticker`: Required, ticker symbol
   - `--date`: Optional, specific date (default: today)
   - `--plot`: Optional, plot type ('3d', 'smile', 'term', 'all')
   - `--save`: Optional, output file path
   - `--metric`: Optional, metric to calculate ('skew', 'term', 'arbitrage', 'all')
   - `--export`: Optional, export metrics to CSV
   - `--interactive`: Flag for plotly interactive plots
   - `--cache`: Flag to use cache
2. [ ] Implement main pipeline function:
   ```python
   def main():
       # Parse arguments
       # Initialize data fetcher
       # Fetch data
       # Process IV
       # Transform coordinates
       # Interpolate surface
       # Generate visualizations (if requested)
       # Calculate metrics (if requested)
       # Export results
   ```
3. [ ] Add error handling and user-friendly messages
4. [ ] Add progress indicators for long operations
5. [ ] Implement logging to file and console
6. [ ] Test CLI with various command combinations:
   ```bash
   python main.py --ticker SPY --plot 3d --save output.png
   python main.py --ticker AAPL --metric all --export metrics.csv
   python main.py --ticker SPY --plot all --interactive
   ```

**Success Criteria**: Can run full pipeline from command line with < 5 commands, generating plots and metrics.

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
2. [ ] Build Streamlit/Plotly Dash dashboard:
   - Interactive ticker selection
   - Real-time surface updates
   - Metric visualization
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

### Week 5: Visualization
- [ ] Module 5: Visualization Suite (complete)
- [ ] Generate all plot types
- [ ] Tests for Module 5

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
- [ ] Dashboard (optional)

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
