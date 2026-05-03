# ₿ Bitcoin Next-Hour Range Predictor
**AlphaI × Polaris Build Challenge**

🔴 **Live App:** https://btc-range-predictor-3abnsrjnjfuhzucsfpucpv.streamlit.app

A Bitcoin price range predictor that forecasts the 95% confidence interval for BTCUSDT's next hourly close. Built on a GBM simulator with GARCH volatility and Student-t fat-tailed innovations, validated against a 30-day backtest of 718 hourly bars.

---

## Overview

Standard volatility models assume constant variance and Gaussian shocks — both false for Bitcoin. This project uses:

- **GARCH(1,1)** captures volatility clustering (calm periods followed by violent ones)
- **Student-t innovations** handle fat tails and extreme moves
- **Cyber-enhanced GBM** amplifies volatility during crisis regimes (high entropy, high absolute returns)
- **Monte Carlo simulation** generates 10,000 price paths per prediction

All predictions are made strictly without future data. The backtest loop only feeds data up to bar N−1 when predicting bar N.

---

## Model Architecture

### 1. Data Fetching
Prices fetched from Binance Vision public API (`data-api.binance.vision`) — no geo-block, no API key needed. Fetches BTCUSDT 1-hour candles with automatic pagination.

### 2. GARCH Volatility Fitting
GARCH(1,1) fitted on log returns scaled by 100, using Student-t error distribution. Captures the empirical observation that volatility shocks in crypto markets cluster together.

- **Inputs:** log return series
- **Outputs:** conditional volatility σ_t, standardised residuals, Student-t degrees of freedom ν

### 3. Cyber-Enhanced GBM Simulation
Each simulated price path follows:
S_t = S_{t-1} × exp((μ − ½σ²)Δt + √(σ²Δt) × Z)
where Z ~ Student-t(ν) scaled to unit variance, and σ² is dynamically adjusted by:

| Signal | Source | Effect |
|--------|--------|--------|
| Shannon Entropy (H) | Rolling 60-bar histogram of residuals | Higher entropy → higher effective volatility |
| Mean Absolute Return (M) | Rolling 60-bar abs log return | Detects momentum/crisis regimes |
| Redundancy | Ratio of short-term to long-term price variance | Amplifies tight, mean-reverting regimes |
| Info Filter | Entropy above its rolling mean | Binary flag that adds 50% variance boost |

A crisis flag (H > 80th pct or M > 80th pct) activates the δ term, further inflating volatility during detected stress periods.

### 4. Monte Carlo Prediction
For a one-step-ahead prediction, 10,000 simulations are run. The 2.5th and 97.5th percentiles form the 95% prediction interval.

---

## Dashboard Features

### Live Prediction
Fetches latest 500 hourly bars from Binance, runs full model pipeline, displays:
- Current BTC price
- Predicted low (2.5th percentile) and high (97.5th percentile) for next hour
- Interval width in USD

### Price Chart with Prediction Ribbon
Shows last 50 bars with green shaded prediction band at the rightmost edge.

### Model Confidence Indicator
| Badge | Condition | Meaning |
|-------|-----------|---------|
| 🟢 High Confidence | Recent vol < 30th percentile | Calm market — reliable intervals |
| 🟡 Normal Confidence | 30th–70th percentile | Normal conditions |
| 🔴 Low Confidence | Recent vol > 70th percentile | High volatility — wider intervals |

### Simulated Price Distribution
Histogram of all 10,000 Monte Carlo simulated next-hour prices with 2.5th/97.5th percentile markers.

### Volatility Regime Chart
Rolling hourly volatility over last 48 bars with calm/volatile threshold lines.

### Prediction History (Part C)
Every dashboard visit saves the current prediction. As hours pass, actual closes are back-filled and each prediction marked ✅ hit / ❌ miss.

---

## Backtest Results
30-day walk-forward backtest over 718 hourly bars (zero data leakage):

| Metric | Value |
|--------|-------|
| Coverage | 96.80% |
| Avg Interval Width | $1,456.42 |
| Mean Winkler Score | $1,783.19 |

---

## Bug Found in Starter
The brief mentions a helper function `evaluate(predictions)` in the starter Colab — it is not present. Metrics were implemented manually using the exact formulas described in the brief.

---

## Project Structure

```
├── app.py                    # Streamlit dashboard: all visualisations and features
├── Copy_of_GBM.ipynb         # Jupyter notebook: data fetch, GARCH fit, GBM simulation, backtest
├── backtest_results.jsonl    # Per-bar backtest output (718 predictions, no data leakage)
├── prediction_history.jsonl  # Live prediction log written by the dashboard
└── requirements.txt          # Pinned Python dependencies
```
---

## Author
**Shreya** | AlphaI × Polaris Build Challenge
