# **Project Specification: 3D Financial Volatility Surface Constructor**

## **1. Executive Summary**

This project involves the development of a robust Python application that constructs, visualizes, and analyzes a three-dimensional volatility surface from live options market data. The surface plots **Implied Volatility (Z-axis)** against **Time to Expiry (X-axis)** and **Moneyness (Y-axis)**, providing a crucial tool for quantitative finance, trading strategy development, and risk assessment. The system will leverage the `yfinance` library for efficient data acquisition while implementing rigorous validation and fallback mechanisms to ensure numerical reliability.

## **2. Project Objectives & Success Criteria**

### **Primary Objectives**
1.  **Automated Data Pipeline**: Create a reliable system to fetch, clean, and structure multi-expiry options chain data for a given underlying asset (e.g., SPY, AAPL).
2.  **Intelligent IV Processing**: Utilize `yfinance`'s provided implied volatility while implementing a robust fallback calculator to fill gaps and validate data quality.
3.  **Surface Construction**: Build a continuous 3D surface via interpolation and extrapolation over a normalized grid of time and moneyness.
4.  **Dynamic Visualization**: Generate interactive, publication-quality 3D plots and 2D slices (volatility smiles/term structure).
5.  **Analytics Engine**: Derive actionable metrics like volatility skew, term structure, and conduct basic arbitrage checks.

### **Key Performance Indicators (KPIs)**
*   **Accuracy**: Calculated IV (when used) matches professional platforms within 0.5% absolute error.
*   **Performance**: Full surface generation from data fetch to plot under 15 seconds.
*   **Robustness**: Successfully processes 95% of trading days, handling missing data gracefully.
*   **Usability**: Clear CLI and optional GUI to generate surfaces with under 5 user commands.

## **3. Technical Architecture**

### **3.1 Technology Stack**
*   **Core Language**: Python 3.9+
*   **Data & Calculation**: `pandas`, `numpy`, `scipy`, `statsmodels`
*   **Visualization**: `matplotlib` (static), `plotly` (interactive), `mayavi` (advanced 3D - optional)
*   **Data Source**: Primary: `yfinance`; Fallback/Validation: Manual Black-Scholes calculation
*   **Development Tools**: `jupyter` for prototyping, `pytest` for testing, `git` for version control.

### **3.2 High-Level System Design**
The application follows a modular, pipeline architecture:
```
User Input (Ticker, Date) 
    → Data Fetcher (yfinance) 
    → Data Validator & IV Enricher 
    → Grid & Coordinate Mapper 
    → Surface Interpolator 
    → Visualization & Analytics Engine 
    → Output (Plots, Metrics, Files)
```

## **4. Detailed Module Specifications**

### **Module 1: SmartOptionsDataFetcher**
**Purpose**: Interface with `yfinance` to retrieve raw options chain data for all available expirations.
*   **Input**: Ticker symbol (string), optional date.
*   **Output**: Dictionary of DataFrames, keyed by expiration date.
*   **Key Logic**: Loops through expiration list from `yfinance`. Implements request throttling and error handling. Caches raw data to disk to avoid repeated API calls during development.
*   **Data Points Retrieved**: `contractSymbol`, `strike`, `lastPrice`, `bid`, `ask`, `volume`, `openInterest`, `impliedVolatility`, `inTheMoney`.

### **Module 2: ImpliedVolatilityProcessor**
**Purpose**: The core reliability module. Ensures a complete, accurate set of implied volatilities.
*   **Input**: Raw DataFrames from Module 1.
*   **Output**: Enhanced DataFrames with a validated `impliedVolatility` column.
*   **Key Logic**:
    1.  **Price Selection**: Defines a "consensus" option price using a hierarchy: `(bid + ask)/2` > `lastPrice`.
    2.  **IV Validation**:
        *   Flags `yfinance` IV values that are negative or excessively high (e.g., > 200%) as suspect.
        *   For suspect or `NaN` values, triggers the fallback calculator.
    3.  **Fallback IV Calculator**: Implements a numerical solver (Newton-Raphson or Bisection) using the Black-Scholes formula to compute IV from the consensus price, spot price, strike, time, and risk-free rate.
    4.  **Risk-Free Rate**: Fetches current yield from `yfinance` (`^TNX` for 10-year) or uses a user-provided constant.

### **Module 3: SurfaceCoordinateEngine**
**Purpose**: Transforms raw option data into a normalized coordinate system suitable for 3D modeling.
*   **Input**: Enhanced DataFrames from Module 2, current spot price.
*   **Output**: Three 1D arrays: `T` (time), `M` (moneyness), `IV` (volatility).
*   **Key Logic**:
    1.  **Time Calculation**: Converts expiration dates to annualized time fractions (e.g., 30/365).
    2.  **Moneyness Calculation**: Converts absolute strike prices to relative moneyness. **Primary Method**: `M = Strike / Spot_Price`. Alternative: `log(Strike/Spot)` can be offered as an option.
    3.  **Filtering**: Removes data points with extreme moneyness (e.g., <0.7 or >1.3) to avoid fitting artifacts from illiquid options.

### **Module 4: VolatilitySurfaceInterpolator**
**Purpose**: Fits a smooth, continuous surface to the discrete, irregularly-spaced (T, M, IV) points.
*   **Input**: Coordinate arrays `T`, `M`, `IV`.
*   **Output**: Dense, regular 2D grid (`T_grid`, `M_grid`, `IV_grid`).
*   **Key Logic**:
    1.  **Interpolation Method**: Implements **2D Cubic Spline** (via `scipy.interpolate.griddata`) as the default for smoothness. Provides **Linear** interpolation as a faster, less smooth alternative.
    2.  **Grid Definition**: Creates a uniform grid spanning the data range (e.g., 50 points in T, 50 in M).
    3.  **Extrapolation Control**: Limits extrapolation range or uses nearest-neighbor to fill edges, preventing wild predictions in areas with no data.

### **Module 5: VisualizationSuite**
**Purpose**: Generates static and interactive visualizations of the surface and its derivatives.
*   **Input**: Dense `T_grid`, `M_grid`, `IV_grid` from Module 4.
*   **Output**: Plot files (PNG, HTML) and displayed figures.
*   **Key Visualizations**:
    1.  **Primary 3D Surface**: Color-mapped surface plot using `matplotlib` 3D axes.
    2.  **Interactive 3D Plot**: Hover-to-see values and rotatable plot using `plotly`.
    3.  **2D Slice - Volatility Smile**: Plot IV vs. Moneyness for 2-3 key expiries (e.g., 30, 90, 180 days).
    4.  **2D Slice - Term Structure**: Plot IV vs. Time for at-the-money (M=1.0) options.

### **Module 6: SurfaceAnalytics**
**Purpose**: Extracts quantitative metrics and insights from the constructed surface.
*   **Input**: Dense surface grid (`T_grid`, `M_grid`, `IV_grid`).
*   **Output**: Computed metrics and diagnostic plots.
*   **Key Functions**:
    1.  **Skew Calculation**: Computes the difference in IV between out-of-the-money puts (e.g., M=0.95) and calls (e.g., M=1.05) for a given expiry.
    2.  **Term Structure**: Returns the array of IVs for at-the-money (M=1.0) across all times.
    3.  **Arbitrage Check (Basic)**: Implements calendar spread arbitrage test: For a given moneyness, IV should be non-decreasing with sqrt(time). Flags violations.
    4.  **Surface Comparison**: Can compute and visualize the difference between two surfaces (e.g., two different dates).

## **5. Project Roadmap & Milestones**

*   **Week 1-2 (Foundation)**: Set up project environment. Complete Modules 1 & 2. Success: Fetch SPY data and output a validated DataFrame with IV for one expiry.
*   **Week 3-4 (Core Logic)**: Develop Modules 3 & 4. Success: Produce a continuous 2D grid of IV from raw data.
*   **Week 5 (Visualization)**: Build Module 5. Success: Generate a static 3D plot and 2D smile slices.
*   **Week 6 (Analytics & Polish)**: Implement Module 6. Create a unified `main.py` CLI. Success: Generate a full report with skew and term structure metrics from a single command.
*   **Week 7 (Testing & Refinement)**: Write unit tests, especially for the IV fallback calculator. Stress-test with various tickers. Document code.
*   **Week 8 (Stretch Goals)**: Explore advanced interpolation (SVI model) or build a simple Streamlit/Plotly Dash web dashboard.

## **6. Deliverables**

1.  **Source Code**: A well-documented, modular Python package as described.
2.  **Command-Line Interface (CLI)**: A `main.py` script allowing commands like:
    ```bash
    python main.py --ticker SPY --plot 3d --save surface_SPY_20231027.png
    python main.py --ticker AAPL --metric skew --export metrics.csv
    ```
3.  **Demonstration Jupyter Notebook**: A step-by-step walkthrough of the library's capabilities.
4.  **Test Suite**: A set of `pytest` scripts ensuring core mathematical and data processing functions are correct.
5.  **Technical Documentation**: Inline docstrings and a `README.md` covering installation, usage, and module overview.

## **7. Risk Mitigation & Assumptions**

*   **Risk**: `yfinance` data is inaccurate or unavailable.
    *   **Mitigation**: The fallback IV calculator ensures core functionality. Log warnings when primary data is used.
*   **Risk**: Interpolation produces unrealistic "bows" or extreme values in sparse data regions.
    *   **Mitigation**: Implement data filtering (Module 3) and use conservative, linear extrapolation at grid edges.
*   **Assumption**: The Black-Scholes model is a sufficient approximation for calculating fallback IV. This is standard practice for this type of project.
*   **Assumption**: The risk-free rate is constant across all maturities. A more sophisticated yield curve can be a stretch goal.

This specification provides a complete blueprint for developing a professional-grade 3D volatility surface constructor, balancing the use of convenient market data with rigorous financial and numerical integrity.