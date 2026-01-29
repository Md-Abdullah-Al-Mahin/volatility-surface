# Implied Volatility Processor: Financial Math Overview

This document explains the financial mathematics behind **Module 2: ImpliedVolatilityProcessor** (`src/iv_processor.py`). It covers how we go from market option prices to implied volatilities used in the volatility surface.

---

## 1. What Problem Are We Solving?

Options data from the market (e.g. yfinance) often includes an **implied volatility (IV)** for each contract. Sometimes that value is missing, wrong (e.g. negative or extreme), or we want to recompute it from prices for consistency.

**Goal:** For each option row we have:

- **Inputs:** Spot price \(S\), strike \(K\), time to expiry \(T\), risk-free rate \(r\), option type (call/put), and a **market price** for the option (e.g. mid of bid/ask or last trade).

- **Output:** The unique **implied volatility** \(\sigma\) such that the Black–Scholes price equals that market price.

So we are solving: **“Given a price, what volatility makes the model match it?”**

---

## 2. Consensus Price (Market Price Input)

We need a single number for “the option’s price” to invert.

- **Preferred:** Mid quote  
  \[
  P_{\text{mid}} = \frac{\text{bid} + \text{ask}}{2}.
  \]
  This is a standard proxy for fair value when both quotes exist.

- **Fallback:** Last traded price, if bid/ask are missing.

- **Otherwise:** We cannot compute IV for that row (no price → no IV).

**Code:** `_calculate_consensus_price(row)` uses `(bid+ask)/2` when both are available, else `lastPrice`, else `NaN`.

---

## 3. Risk-Free Rate \(r\)

Option pricing formulas use a **continuously compounded** risk-free rate \(r\) (decimal, e.g. 0.05 = 5%).

- **Source:** We try to use the 10-year Treasury yield (e.g. ^TNX from yfinance). Treasury yields are quoted in **percent** (e.g. 4.5), so we use  
  \[
  r = \frac{\text{yield (\%)}}{100}.
  \]

- **Fallback:** A user-provided constant or default 0.05.

**Code:** `_get_risk_free_rate()` fetches ^TNX and divides by 100, or uses the configured constant.

---

## 4. Time to Expiry \(T\)

All formulas use **time in years**.

- **Definition:**  
  \[
  T = \frac{\text{(expiration date)} - \text{(today)}}{\text{1 year}}.
  \]

- **Convention:** We use **calendar days / 365** (act/365):
  \[
  T = \frac{(\text{exp\_date} - \text{today}).\text{days}}{365}.
  \]

- **Floor:** We use \(T = \max(\ldots, 10^{-6})\) so we never pass \(T \le 0\) into Black–Scholes (which would be undefined).

**Code:** In `process_iv`, \(T\) is computed from the DataFrame’s `expirationDate` (or the dictionary key) and `date.today()`, then converted to years with a small lower bound.

---

## 5. Black–Scholes Formula (European Options)

We assume **no dividends** and **European** exercise (exercise only at expiry). The standard formulas are below; our code matches them.

### 5.1 Auxiliary quantities \(d_1\) and \(d_2\)

\[
d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T} = \frac{\ln(S/K) + (r - \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}.
\]

- \(S\): spot price  
- \(K\): strike  
- \(T\): time to expiry (years)  
- \(r\): risk-free rate (decimal)  
- \(\sigma\): volatility (decimal)

**Code check:**  
`d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)`  
`d2 = d1 - sigma*sqrt_T`  
✓ Matches the formulas above.

### 5.2 Call price

\[
C = S\,N(d_1) - K\,e^{-rT}\,N(d_2).
\]

\(N(\cdot)\) is the **standard normal CDF** (cumulative distribution function).

**Code check:**  
`S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)`  
✓ Correct.

### 5.3 Put price

\[
P = K\,e^{-rT}\,N(-d_2) - S\,N(-d_1).
\]

**Code check:**  
`K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)`  
✓ Correct.

### 5.4 Put–call parity (sanity check)

\[
C - P = S - K\,e^{-rT}.
\]

So if both call and put formulas are correct, \(C - P\) must equal \(S - K\,e^{-rT}\). Our tests include a check that \(|(C-P) - (S - K\,e^{-rT})|\) is negligible.

### 5.5 Edge cases

- If \(T \le 0\) or \(\sigma \le 0\), the expressions for \(d_1,d_2\) are not well-defined. The code returns `NaN` in those cases. ✓

---

## 6. Vega (Sensitivity to Volatility)

To solve “price = market price” for \(\sigma\), we use **Newton–Raphson**. That requires the derivative of the option price with respect to \(\sigma\), which is **vega**.

**Standard formula (call and put):**

\[
\text{Vega} = \frac{\partial V}{\partial \sigma} = S\,\sqrt{T}\,n(d_1),
\]

where \(n(x) = N'(x) = \frac{1}{\sqrt{2\pi}}\,e^{-x^2/2}\) is the **standard normal PDF**.

**Code check:**  
`S * sqrt_T * norm.pdf(d1)`  
✓ Correct (same \(d_1\) as in the price formula).

---

## 7. Solving for Implied Volatility

We want \(\sigma\) such that

\[
V(\sigma) = P_{\text{market}},
\]

where \(V(\sigma)\) is the Black–Scholes price (call or put) for volatility \(\sigma\).

### 7.1 Newton–Raphson

We solve \(f(\sigma) = V(\sigma) - P_{\text{market}} = 0\).

Update:

\[
\sigma_{n+1} = \sigma_n - \frac{f(\sigma_n)}{f'(\sigma_n)} = \sigma_n - \frac{V(\sigma_n) - P_{\text{market}}}{\text{Vega}(\sigma_n)}.
\]

- **Code:** `diff = price - market_price` (i.e. \(V(\sigma_n) - P_{\text{market}}\)), then `sigma = sigma - diff / vega`.  
  So we are doing \(\sigma - \frac{\text{diff}}{\text{vega}} = \sigma - \frac{V - P_{\text{market}}}{\text{Vega}}\). ✓ Correct.

- **Initial guess:** \(\sigma_0 = 0.2\) (20%) is a common choice.

- **Safeguards:** If \(\sigma\) goes negative we clamp to a tiny positive value; if it explodes we reset. If vega is too small (numerically zero), we stop and fall back to bisection.

### 7.2 Bisection (fallback)

- **Idea:** \(V(\sigma)\) is **increasing** in \(\sigma\) for both calls and puts (higher vol → higher option value). So there is at most one \(\sigma\) with \(V(\sigma) = P_{\text{market}}\).

- **Bracket:** We need \(\sigma_{\text{low}}\) and \(\sigma_{\text{high}}\) such that  
  \(V(\sigma_{\text{low}}) \le P_{\text{market}} \le V(\sigma_{\text{high}})\).  
  Code uses `low=1e-4`, `high=3.0` (0.01% and 300% vol).

- **Step:** Let \(\sigma_{\text{mid}} = (\sigma_{\text{low}} + \sigma_{\text{high}})/2\).  
  - If \(V(\sigma_{\text{mid}}) > P_{\text{market}}\), the true \(\sigma\) is smaller → set \(\sigma_{\text{high}} = \sigma_{\text{mid}}\).  
  - If \(V(\sigma_{\text{mid}}) < P_{\text{market}})\), the true \(\sigma\) is larger → set \(\sigma_{\text{low}} = \sigma_{\text{mid}}\).  
  Repeat until \(|\sigma_{\text{high}} - \sigma_{\text{low}}|\) (or price error) is small enough.

**Code check:**  
- `if pmid > market_price: high = mid` (price too high → reduce \(\sigma\)) ✓  
- `if pmid < market_price: low = mid` (price too low → increase \(\sigma\)) ✓  

Bisection is robust when Newton–Raphson fails (e.g. bad vega or poor initial guess).

---

## 8. IV Validation

Not every number we get from the solver is acceptable:

- **Negative IV** is meaningless → reject.
- **Extremely high IV** (e.g. > 200%) is often a sign of bad data or illiquid contract → reject.
- **NaN** (e.g. missing price, \(T\le 0\)) → reject.

**Code:** `_validate_iv(iv)` returns true only if \(0 \le \text{IV} \le 2.0\) (and not NaN). ✓

---

## 9. Summary: Flow in Code

| Step | Math | Code |
|------|------|------|
| Market price | \((bid+ask)/2\) or lastPrice | `_calculate_consensus_price` |
| Risk-free rate | \(r\) from ^TNX/100 or constant | `_get_risk_free_rate` |
| Time to expiry | \(T = \text{days}/365\), \(T\ge 10^{-6}\) | In `process_iv` from `expirationDate` |
| Option price | \(C\) or \(P\) via \(d_1,d_2\), \(N(d_1),N(d_2)\) | `_black_scholes_price` |
| Vega | \(S\sqrt{T}\,n(d_1)\) | `_black_scholes_vega` |
| IV solve | Newton: \(\sigma - (V-P)/\text{Vega}\); else bisection | `_calculate_iv_newton_raphson`, `_calculate_iv_bisection` |
| Sanity | \(0 \le \sigma \le 2\) | `_validate_iv` |

---

## 10. Math Verification Checklist

| Item | Formula / fact | Code | Status |
|------|----------------|------|--------|
| \(d_1\) | \(\frac{\ln(S/K)+(r+\sigma^2/2)T}{\sigma\sqrt{T}}\) | `(np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)` | ✓ |
| \(d_2\) | \(d_1 - \sigma\sqrt{T}\) | `d1 - sigma*sqrt_T` | ✓ |
| Call | \(S N(d_1) - K e^{-rT} N(d_2)\) | `S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)` | ✓ |
| Put | \(K e^{-rT} N(-d_2) - S N(-d_1)\) | `K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)` | ✓ |
| Vega | \(S\sqrt{T}\,n(d_1)\) | `S*sqrt_T*norm.pdf(d1)` | ✓ |
| Newton step | \(\sigma - (V - P_{\text{market}})/\text{Vega}\) | `sigma - diff/vega` | ✓ |
| Bisection | Price increasing in \(\sigma\); bracket and halve | `pmid > market_price → high=mid` etc. | ✓ |
| \(T\) | Years, \(\ge 10^{-6}\) | `max(..., 1e-6)` | ✓ |
| Consensus | Mid or last | `(bid+ask)/2` or `lastPrice` | ✓ |
| IV bounds | \([0, 2]\) | `_validate_iv`: 0 ≤ iv ≤ 2, not NaN | ✓ |

All mathematical steps in the IV processor match the standard Black–Scholes setup and common numerical methods for inverting the price to get implied volatility.
