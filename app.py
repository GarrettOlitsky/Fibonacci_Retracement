import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================
# TRON Neon Theme Injection
# =========================
PRIMARY = "#00ffe0"   # neon cyan
ACCENT  = "#00ff88"   # neon green
BG      = "#0b0f1a"   # deep space
PLOTBG  = "#0f1324"   # slightly lighter panel
GRID    = "#19324a"   # teal-ish grid

def inject_neon_theme():
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            html, body, [data-testid="stAppViewContainer"] {{
                background: radial-gradient(1200px 800px at 20% -10%, #101735 0%, {BG} 45%) fixed;
                color: #cde7ff !important;
                font-family: 'Orbitron', sans-serif;
            }}
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #0b0f1a 0%, #0b0f1a 70%, #0a0e19 100%) !important;
                border-right: 1px solid #0f2340;
            }}
            h1, h2, h3 {{
                color: {PRIMARY} !important;
                text-shadow: 0 0 12px rgba(0,255,224,.3), 0 0 24px rgba(0,255,224,.15);
            }}
            [data-testid="stMetricValue"] {{
                color: {ACCENT} !important;
                text-shadow: 0 0 10px rgba(0,255,136,.5);
            }}
            [data-testid="stMetricLabel"] {{
                color: #9dc6ff !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Streamlit Config
# =========================
st.set_page_config(
    page_title="Fibonacci Day Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_neon_theme()

# =========================
# Data Helpers
# =========================
INTERVAL_TO_PERIOD = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "60m": "730d", "1h": "730d", "1d": "max",
}
def _normalize_interval(interval: str) -> str:
    return "60m" if interval == "1h" else interval

@st.cache_data(show_spinner=False)
def fetch(symbol: str, interval: str) -> pd.DataFrame:
    yf_interval = _normalize_interval(interval)
    period = INTERVAL_TO_PERIOD.get(interval, "60d")
    df = yf.download(symbol, period=period, interval=yf_interval, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df = df.rename(columns=str.title)
    return df[["Date","Open","High","Low","Close","Volume"]]

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.rolling(length).mean()
    roll_dn = loss.rolling(length).mean().replace(0, np.nan)
    rs = roll_up / roll_dn
    return (100 - (100 / (1 + rs))).bfill()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def vwap(df: pd.DataFrame) -> pd.Series:
    day = pd.to_datetime(df["Date"]).dt.date
    pv = df["Close"] * df["Volume"]
    return (pv.groupby(day).cumsum() / df["Volume"].groupby(day).cumsum().replace(0,np.nan)).bfill()

def recent_swing(df: pd.DataFrame, lookback: int = 200):
    n = min(len(df), lookback)
    if n < 5: return None, None, None
    sub = df.tail(n).reset_index(drop=True)
    hi_pos = int(np.argmax(sub["High"].to_numpy()))
    lo_pos = int(np.argmin(sub["Low"].to_numpy()))
    hi = (float(sub.loc[hi_pos,"High"]), sub.loc[hi_pos,"Date"])
    lo = (float(sub.loc[lo_pos,"Low"]),  sub.loc[lo_pos,"Date"])
    return hi, lo, (lo_pos < hi_pos)

def fib_levels(high: float, low: float, trend_up: bool):
    vals = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    labels = ["0%","23%","38%","50%","61%","79%","100%"]
    lvl = {}
    if trend_up:
        for L, f in zip(labels, vals): lvl[L] = high - (high-low)*f
    else:
        for L, f in zip(labels, vals): lvl[L] = low  + (high-low)*f
    return lvl

def nearest_levels(price: float, levels: dict, n=3):
    items = sorted(levels.items(), key=lambda kv: abs(kv[1]-price))[:n]
    return [{"Level":k, "Price":round(v,4), "Δ%":round((price-v)/price*100,3)} for k,v in items]

# =========================
# Plotting
# =========================
def plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", opacity=0.95
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor=BG, plot_bgcolor=PLOTBG,
        font=dict(color="#cde7ff", family="Orbitron"),
        height=600, margin=dict(l=20,r=20,t=50,b=30),
        xaxis_rangeslider_visible=False,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showline=True, linecolor="#1e2b45"),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, showline=True, linecolor="#1e2b45"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#15304a"),
    )
    return fig

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Symbol", value="VOO").upper().strip()
    tf = st.selectbox("Primary Timeframe", ["1m","5m","15m","1h","1d"], index=2)
    lookback = st.slider("Swing Lookback (candles)", 50, 500, 120, step=10)
    show_vwap = st.checkbox("VWAP", value=True)
    show_rsi  = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD (12,26,9)", value=False)

# =========================
# Data
# =========================
df = fetch(symbol, tf)
if df.empty:
    st.warning("No data returned. Try another symbol/timeframe.")
    st.stop()

df["RSI"] = rsi(df["Close"], 14)
mline, msig, mhist = macd(df["Close"])
df["MACD"], df["MACD_sig"], df["MACD_hist"] = mline, msig, mhist
df["VWAP"] = vwap(df)
df["MA20"] = df["Close"].rolling(20).mean()

hi, lo, trend_up = recent_swing(df, lookback)
if hi is None or lo is None:
    st.warning("Not enough candles to detect a swing.")
    st.stop()

fib_primary = fib_levels(hi[0], lo[0], trend_up)
price = float(df["Close"].iloc[-1])

# =========================
# Chart
# =========================
title = f"{symbol} — {tf} — {price:.2f}"
fig = plot_candles(df.tail(300), title)

# Add fib lines (legend hidden)
for name, level in fib_primary.items():
    fig.add_hline(
        y=level,
        line=dict(width=1, dash="dot", color=PRIMARY),
        annotation_text=f"{name}: {level:.2f}",
        annotation_font=dict(color=PRIMARY, family="Orbitron"),
        annotation_bgcolor="rgba(0,0,0,0.35)",
        annotation_position="right",
        opacity=0.85,
    )

# Indicators
if show_vwap:
    fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP",
                             line=dict(width=1.4, color=ACCENT)))
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20",
                         line=dict(width=1.2, color="#6fb1ff")))

st.plotly_chart(fig, use_container_width=True)

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Price", f"{price:.2f}")
if show_rsi:  c2.metric("RSI (14)", f"{float(df['RSI'].iloc[-1]):.1f}")
if show_macd: c3.metric("MACD", f"{float(df['MACD'].iloc[-1]):.2f}")
