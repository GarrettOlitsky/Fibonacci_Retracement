import time
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================
# App Config
# =========================
st.set_page_config(page_title="Fibonacci Day Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# =========================
# Helpers
# =========================
INTERVAL_TO_PERIOD = {
    "1m": "7d",     # yfinance limit
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h": "730d",   # we'll map to 60m
    "1d": "max",
}

def _normalize_interval(interval: str) -> str:
    return "60m" if interval == "1h" else interval

@st.cache_data(show_spinner=False)
def fetch(symbol: str, interval: str) -> pd.DataFrame:
    """Download OHLCV with yfinance, normalized columns."""
    yf_interval = _normalize_interval(interval)
    period = INTERVAL_TO_PERIOD.get(interval, "60d")
    df = yf.download(symbol, period=period, interval=yf_interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    # Normalize datetime column to "Date"
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df = df.rename(columns=str.title)
    return df

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.rolling(length).mean()
    roll_dn = loss.rolling(length).mean().replace(0, np.nan)
    rs = roll_up / roll_dn
    out = 100 - (100 / (1 + rs))
    return out.bfill()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def vwap(df: pd.DataFrame) -> pd.Series:
    # Reset daily
    day = pd.to_datetime(df["Date"]).dt.date
    pv = df["Close"] * df["Volume"]
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).bfill()

def recent_swing(df: pd.DataFrame, lookback: int = 200):
    """
    Pick recent swing high/low from last N candles.
    Returns (hi_price, hi_time), (lo_price, lo_time), trend_up(bool).
    """
    n = min(len(df), lookback)
    if n < 5:
        return None, None, None
    sub = df.tail(n)
    hi_idx = sub["High"].idxmax()
    lo_idx = sub["Low"].idxmin()
    hi = (float(df.loc[hi_idx, "High"]), df.loc[hi_idx, "Date"])
    lo = (float(df.loc[lo_idx, "Low"]),  df.loc[lo_idx, "Date"])
    trend_up = lo_idx < hi_idx
    return hi, lo, trend_up

def fib_levels(high: float, low: float, trend_up: bool):
    levels_raw = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    labels = ["0%", "23%", "38%", "50%", "61%", "79%", "100%"]
    lvls = {}
    if trend_up:
        for lbl, l in zip(labels, levels_raw):
            lvls[lbl] = high - (high - low) * l
    else:
        for lbl, l in zip(labels, levels_raw):
            lvls[lbl] = low + (high - low) * l
    return lvls

def nearest_levels(price: float, levels: dict, n=3):
    items = sorted(levels.items(), key=lambda kv: abs(kv[1] - price))
    out = []
    for k, v in items[:n]:
        out.append({"Level": k, "Price": round(v, 4), "Δ%": round((price - v) / price * 100, 3)})
    return out

def confluence_from_timeframes(symbol: str, tfs: list[str], lookback: int = 200, tol_pct: float = 0.2):
    rows = []
    for tf in tfs:
        dfe = fetch(symbol, tf)
        if dfe.empty:
            continue
        hi, lo, up = recent_swing(dfe, lookback)
        if hi is None or lo is None:
            continue
        fib = fib_levels(hi[0], lo[0], trend_up=up)
        for k, v in fib.items():
            rows.append({"tf": tf, "level": k, "price": float(v)})
    if not rows:
        return [], []
    rows = sorted(rows, key=lambda r: r["price"])
    clusters, group, base = [], [], None
    for r in rows:
        if base is None:
            base = r["price"]; group = [r]
        elif abs(r["price"] - base) / base * 100.0 <= tol_pct:
            group.append(r)
        else:
            clusters.append(group)
            base = r["price"]; group = [r]
    if group:
        clusters.append(group)
    # summarize
    summaries = []
    for g in clusters:
        lo_p = min(x["price"] for x in g)
        hi_p = max(x["price"] for x in g)
        summaries.append({"y0": round(lo_p, 4), "y1": round(hi_p, 4), "count": len(g), "members": g})
    return rows, summaries

def plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", opacity=0.95
    ))
    fig.update_layout(
        title=title,
        height=620,
        margin=dict(l=20, r=20, t=40, b=30),
        xaxis_rangeslider_visible=False
    )
    return fig

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Symbol", value="VOO").upper().strip()
    tf = st.selectbox("Primary Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=2)
    extra_tfs = st.multiselect("Confluence TFs", ["1m", "5m", "15m", "1h", "1d"], default=["1h", "1d"])
    lookback = st.slider("Swing Lookback (candles)", min_value=50, max_value=500, value=200, step=10)
    show_vwap = st.checkbox("VWAP", value=True)
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD (12,26,9)", value=False)
    alert_level = st.selectbox("Alert on primary Fib", ["Off","23%","38%","50%","61%","79%"], index=0)
    refresh_sec = st.number_input("Auto-refresh (seconds)", min_value=0, max_value=300, value=0, step=1)
    st.caption("Tip: set refresh to 5–15s for active monitoring.")

# optional auto-refresh
if refresh_sec and refresh_sec > 0:
    st_autorefresh = st.experimental_rerun  # placeholder to please linters
    st.runtime.legacy_caching.clear_cache()  # keep data fresh if desired
    st.experimental_set_query_params(ts=int(time.time()))

# =========================
# Data
# =========================
df = fetch(symbol, tf)
if df.empty:
    st.warning("No data returned. Try another symbol/timeframe.")
    st.stop()

# Indicators
df["RSI"] = rsi(df["Close"], 14)
macd_line, macd_sig, macd_hist = macd(df["Close"])
df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd_line, macd_sig, macd_hist
df["VWAP"] = vwap(df)
df["MA20"] = df["Close"].rolling(20).mean()

hi, lo, trend_up = recent_swing(df, lookback)
if hi is None or lo is None:
    st.warning("Not enough candles to detect a swing.")
    st.stop()

fib_primary = fib_levels(hi[0], lo[0], trend_up=trend_up)
price = float(df["Close"].iloc[-1])

# Confluence
_, clusters = confluence_from_timeframes(symbol, [t for t in extra_tfs if t != tf], lookback=lookback, tol_pct=0.2)

# =========================
# Main Chart
# =========================
title = f"{symbol} — {tf} — {price:.2f}"
fig = plot_candles(df.tail(300), title)

# Primary fib lines
for name, level in fib_primary.items():
    fig.add_hline(y=level, line=dict(width=1, dash="dot"), annotation_text=f"{name}: {level:.2f}",
                  annotation_position="right", opacity=0.6)

# Confluence shading
for c in clusters:
    fig.add_hrect(y0=c["y0"], y1=c["y1"], line_width=0, fillcolor="LightGreen", opacity=0.13)

# Indicators on-price
if show_vwap:
    fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP", opacity=0.9))
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20", opacity=0.6))

st.plotly_chart(fig, use_container_width=True)

# Indicator readouts
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Price", f"{price:.2f}")
with col2:
    last_rsi = float(df["RSI"].iloc[-1])
    if show_rsi:
        st.metric("RSI (14)", f"{last_rsi:.1f}")
with col3:
    if show_macd:
        st.metric("MACD", f"{float(df['MACD'].iloc[-1]):.3f}")
with col4:
    if show_macd:
        st.metric("Signal", f"{float(df['MACD_sig'].iloc[-1]):.3f}")

# Nearest levels table
st.subheader("Nearest Primary Fib Levels")
near = nearest_levels(price, fib_primary, n=3)
st.table(pd.DataFrame(near))

# =========================
# Alerts
# =========================
if alert_level != "Off":
    key = alert_level
    target = fib_primary.get(key)
    if target is not None and len(df) >= 2:
        price_prev = float(df["Close"].iloc[-2])
        touched = abs(price - target) / price <= 0.001  # within 0.1%
        broken_up = price > target and price_prev <= target
        broken_dn = price < target and price_prev >= target
        if touched or broken_up or broken_dn:
            st.success(f"🔔 {symbol}: {key} @ {target:.2f} touched/broken on {tf}")
        else:
            st.info(f"Alert armed: {symbol} {key} @ {target:.2f} (refresh to check)")

# =========================
# Position Sizing
# =========================
st.markdown("### Position Sizing")
pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
with pc2:
    entry = st.number_input("Entry", value=float(price), step=0.01, format="%.2f")
with pc3:
    # default stop at nearest fib level
    default_stop = float(near[0]["Price"]) if near else price * 0.99
    stop = st.number_input("Stop", value=default_stop, step=0.01, format="%.2f")
with pc4:
    acct = st.number_input("Account Size ($)", value=10000, step=100)

risk_amt = acct * (risk_pct / 100.0)
per_share = abs(entry - stop)
shares = int(risk_amt / per_share) if per_share > 0 else 0
rr = (fib_primary.get("0%", entry) - entry) / (entry - stop) if (entry - stop) != 0 else np.nan

cols = st.columns(3)
cols[0].write(f"**Max Position**: `{shares}` shares")
cols[1].write(f"**Risk Amount**: `${risk_amt:.2f}`")
cols[2].write(f"**Est. R:R to 0%**: `{rr:.2f}`")

# =========================
# Secondary Panels
# =========================
with st.expander("Confluence Details"):
    if clusters:
        st.write(pd.DataFrame([
            {
                "Zone Low": c["y0"],
                "Zone High": c["y1"],
                "Count": c["count"],
                "Members": ", ".join([f"{m['tf']} {m['level']}" for m in c["members"]])
            } for c in clusters
        ]))
    else:
        st.write("No confluence clusters found for selected timeframes.")

with st.expander("Notes / How to Use"):
    st.markdown("""
- **Draw logic**: App auto-detects the most recent swing (low→high = up-swing; high→low = down-swing) and plots standard fib retracements (23/38/50/61/79%).
- **Confluence**: Shaded zones show where fib levels from other timeframes cluster (±0.2%). These are prime **reaction areas**.
- **Confirmation**: Combine fibs with **VWAP / RSI / MACD**. Example: long if pullback to 38–61% **and** RSI not overbought **and** price above VWAP.
- **Risk**: Use the built-in position sizer. Place stops beyond the level you’re trading against (e.g., just below 61% in an uptrend).
- **Refresh**: Set auto-refresh for live monitoring during the session.
- **Reminder**: This is a **tool**, not a signal. Always validate with your plan.
""")
