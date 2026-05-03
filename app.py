import streamlit as st
import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model
import json, os, time
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="BTC Next-Hour Range Predictor", layout="wide", page_icon="₿")

HISTORY_FILE = "prediction_history.jsonl"

@st.cache_data(ttl=300)
def get_btc_hourly(n_bars=600):
    url = "https://data-api.binance.vision/api/v3/klines"
    all_bars = []
    end_time = None
    while len(all_bars) < n_bars:
        params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 1000}
        if end_time:
            params["endTime"] = end_time
        r = requests.get(url, params=params, timeout=15)
        bars = r.json()
        if not bars:
            break
        all_bars = bars + all_bars
        end_time = bars[0][0] - 1
        if len(bars) < 1000 or len(all_bars) >= n_bars:
            break
        time.sleep(0.1)
    all_bars = all_bars[-n_bars:]
    df = pd.DataFrame(all_bars, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_vol','trades','tb','tq','ignore'])
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df.set_index('close_time', inplace=True)
    return df['close'].astype(float)

def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

def fit_and_predict(prices, n_sims=5000):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    am = arch_model(log_ret * 100, vol='Garch', p=1, q=1, dist='studentst')
    res = am.fit(disp='off', show_warning=False)
    sig = res.conditional_volatility / 100
    resid = (log_ret * 100 - res.params['mu']) / res.conditional_volatility
    nu = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])
    H = rolling_entropy(resid)
    M = log_ret.abs().rolling(60).mean()
    bar_s2 = (sig**2).mean()
    red = (1 + 0.1 * np.log1p(
        prices.rolling(5).var() / prices.rolling(20).var()
    )).fillna(1)
    inf_f = (H > H.mean()).fillna(False).astype(float)
    H_max = H.max() if H.max() > 0 else 1.0
    M_max = M.max() if M.max() > 0 else 1.0
    a0, d0 = 0.5, 0.3
    if H_max > 0 and M_max > 0 and a0*H_max + d0*M_max >= 1:
        fac = 0.95 / (a0*H_max + d0*M_max)
        a0 *= fac; d0 *= fac
    S0 = float(prices.iloc[-1])
    s2_last = float(sig.iloc[-1])**2
    H_val = min(float(H.iloc[-1])/H_max, 1.0) if not np.isnan(H.iloc[-1]) else 0.0
    M_val = min(float(M.iloc[-1])/M_max, 1.0) if not np.isnan(M.iloc[-1]) else 0.0
    crisis = (H_val > 0.8) or (M_val > 0.8)
    dt_c = d0 if crisis else 0.0
    sigma2 = (sig.iloc[-1]**2 * (1 + a0*H_val + dt_c*M_val)
              + 0.2*(bar_s2 - s2_last))
    sigma2 *= max(1e-12, float(red.iloc[-1]))
    sigma2 *= 1 + 0.5*float(inf_f.iloc[-1])
    sigma2 = max(1e-8, min(sigma2, 0.5))
    mu = float(log_ret.mean())
    Z = np.random.standard_t(nu, size=n_sims) * np.sqrt((nu-2)/nu)
    S1 = S0 * np.exp((mu - 0.5*sigma2) + np.sqrt(sigma2)*Z)
    low95, high95 = np.percentile(S1, [2.5, 97.5])
    vol_pct = float(sig.iloc[-1]) * 100
    vol_30 = float(sig.quantile(0.30)) * 100
    vol_70 = float(sig.quantile(0.70)) * 100
    if vol_pct < vol_30:
        regime = "Calm"
    elif vol_pct > vol_70:
        regime = "Volatile"
    else:
        regime = "Normal"
    return S0, low95, high95, S1, sig, regime

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]

def save_prediction(S0, low95, high95, next_bar_time, prices_recent):
    history = load_history()
    for rec in history:
        if rec.get('actual') is None:
            t = pd.Timestamp(rec['predicted_for'])
            matches = prices_recent.index[prices_recent.index >= t]
            if len(matches) > 0:
                rec['actual'] = float(prices_recent[matches[0]])
                rec['hit'] = int(rec['low_95'] <= rec['actual'] <= rec['high_95'])
    new_rec = {
        "predicted_for": str(next_bar_time),
        "S0": S0,
        "low_95": low95,
        "high_95": high95,
        "actual": None,
        "hit": None,
        "saved_at": str(datetime.utcnow())
    }
    history.append(new_rec)
    with open(HISTORY_FILE, 'w') as f:
        for r in history:
            f.write(json.dumps(r) + '\n')
    return history

# ── Load backtest metrics ──────────────────────────────────────────────────
COVERAGE, AVG_WIDTH, MEAN_WINKLER, N_PREDS = 0.95, 0.0, 0.0, 0
if os.path.exists("backtest_results.jsonl"):
    bt = []
    with open("backtest_results.jsonl") as f:
        for l in f:
            if l.strip():
                bt.append(json.loads(l))
    if bt:
        COVERAGE     = sum(r['hit'] for r in bt) / len(bt)
        AVG_WIDTH    = sum(r['width'] for r in bt) / len(bt)
        MEAN_WINKLER = sum(r['winkler'] for r in bt) / len(bt)
        N_PREDS      = len(bt)

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("₿ Bitcoin Next-Hour Range Predictor")
st.caption("GBM Monte Carlo model — BTCUSDT 1h candles via Binance")

# Backtest metrics
st.subheader("Backtest Performance (30-day)")
c1, c2, c3 = st.columns(3)
c1.metric("Coverage (95%)", f"{COVERAGE:.2%}")
c2.metric("Avg Width", f"${AVG_WIDTH:,.2f}")
c3.metric("Mean Winkler", f"${MEAN_WINKLER:,.2f}")
st.caption(f"{N_PREDS} predictions · local backtest.py")
st.divider()

# Live prediction
with st.spinner("Fetching live BTC data and running model..."):
    prices = get_btc_hourly(500)
    S0, low95, high95, S1, sig_series, regime = fit_and_predict(prices)
    next_bar = prices.index[-1] + pd.Timedelta(hours=1)

history = save_prediction(S0, low95, high95, next_bar, prices)

st.subheader("Live Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Current BTC Price", f"${S0:,.2f}")
c2.metric("95% CI Lower Bound", f"${low95:,.2f}")
c3.metric("95% CI Upper Bound", f"${high95:,.2f}")

st.info(f"**95% Prediction Interval (US Dollar):** ${low95:,.2f} — ${high95:,.2f} (width: ${high95-low95:,.2f})")

conf_color = "🟢" if regime == "Calm" else "🟡" if regime == "Normal" else "🔴"
conf_msg = {"Calm": "Recent volatility is low — model predictions are more reliable.",
            "Normal": "Moderate volatility — normal prediction confidence.",
            "Volatile": "High volatility — ranges may be wider than usual."}
st.success(f"**Model Confidence:** {conf_color} **{regime} Confidence** — {conf_msg[regime]}")

# Price chart
st.subheader("Price Chart with Prediction Range")
last50 = prices.iloc[-50:]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=last50.index, y=last50.values,
    name="BTC Close", line=dict(color='cyan', width=1.5)))
fig.add_trace(go.Scatter(
    x=[last50.index[-1], next_bar],
    y=[low95, low95], mode='lines',
    line=dict(color='rgba(0,200,0,0.5)', dash='dash'), showlegend=False))
fig.add_trace(go.Scatter(
    x=[last50.index[-1], next_bar],
    y=[high95, high95], mode='lines',
    fill='tonexty', fillcolor='rgba(0,200,0,0.2)',
    line=dict(color='rgba(0,200,0,0.5)', dash='dash'), name="95% Band"))
fig.add_trace(go.Scatter(
    x=[next_bar], y=[(low95+high95)/2],
    mode='markers', marker=dict(color='lime', size=12, symbol='diamond'),
    name="Next-Hour Range"))
fig.add_annotation(x=next_bar, y=high95, text=f"${high95:,.2f}", showarrow=False, yshift=10)
fig.add_annotation(x=next_bar, y=low95, text=f"${low95:,.2f}", showarrow=False, yshift=-15)
fig.update_layout(
    height=400, xaxis_title="Time (IST)",
    yaxis_title="Price (USD)",
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation='h', yanchor='bottom', y=1.02))
st.plotly_chart(fig, use_container_width=True)

# Simulated distribution
st.subheader("Simulated Next-Hour Price Distribution")
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=S1, nbinsx=50, marker_color='steelblue', opacity=0.8))
fig2.add_vline(x=low95, line_dash='dash', line_color='red',
               annotation_text=f"5th pct: ${low95:,.2f}", annotation_position="top left")
fig2.add_vline(x=S0, line_dash='dot', line_color='white',
               annotation_text=f"Current: ${S0:,.2f}")
fig2.add_vline(x=high95, line_dash='dash', line_color='red',
               annotation_text=f"95th pct: ${high95:,.2f}", annotation_position="top right")
fig2.update_layout(
    height=300, xaxis_title="Simulated Close Price (USD)", yaxis_title="Frequency",
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig2, use_container_width=True)

# Volatility regime
st.subheader(f"Volatility Regime — Current: {regime}")
last48_sig = sig_series.iloc[-48:] * 100
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=last48_sig.index, y=last48_sig.values,
    name="Hourly Volatility (%)", line=dict(color='violet', width=1.5)))
fig3.add_hline(y=float(sig_series.quantile(0.70))*100,
               line_dash='dash', line_color='red',
               annotation_text="70th pct — Volatile threshold")
fig3.add_hline(y=float(sig_series.quantile(0.30))*100,
               line_dash='dash', line_color='green',
               annotation_text="30th pct — Calm threshold")
fig3.update_layout(
    height=250, xaxis_title="Time (IST)", yaxis_title="Hourly Volatility (%)",
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig3, use_container_width=True)

# Part C — Prediction History
st.subheader("Prediction History")
st.caption("Predictions are saved on each visit. Actuals fill in as candles close.")
filled = [r for r in history if r.get('actual') is not None]
if filled:
    hits = sum(r['hit'] for r in filled)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predictions with Actuals", len(filled))
    c2.metric("Hits", f"{hits} / {len(filled)}")
    c3.metric("Live Accuracy", f"{hits/len(filled):.1%}")
    rows = []
    for i, r in enumerate(sorted(filled, key=lambda x: x['predicted_for'], reverse=True), 1):
        rows.append({
            "#": i,
            "Predicted For (IST)": r['predicted_for'],
            "Price at Prediction (USD)": f"${r['S0']:,.2f}",
            "Low 5% (USD)": f"${r['low_95']:,.2f}",
            "High 95% (USD)": f"${r['high_95']:,.2f}",
            "Actual Close (USD)": f"${r['actual']:,.2f}",
            "Result": "✅" if r['hit'] else "❌"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No completed predictions yet. Check back after the next hour closes.")

st.divider()
st.caption("Built by Shreya | AlphaI × Polaris Build Challenge")