import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================
# Neon / TRON theme helpers
# =========================
PRIMARY = "#00ffe0"   # neon cyan
ACCENT  = "#00ff88"   # neon green
BG      = "#0b0f1a"   # deep space
PLOTBG  = "#0f1324"   # slightly lighter panel
GRID    = "#19324a"   # teal-ish grid

def inject_neon_theme():
    # Google font + CSS overrides
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: {BG};
                --panel: {PLOTBG};
                --primary: {PRIMARY};
                --accent: {ACCENT};
                --grid: {GRID};
                --text: #cde7ff;
            }}
            html, body, [data-testid="stAppViewContainer"] {{
                background: radial-gradient(1200px 800px at 20% -10%, #101735 0%, var(--bg) 45%) fixed;
                color: var(--text) !important;
                font-family: 'Orbitron', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif;
            }}
            /* Sidebar */
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #0b0f1a 0%, #0b0f1a 70%, #0a0e19 100%) !important;
                border-right: 1px solid #0f2340;
                box-shadow: inset -6px 0 16px rgba(0,0,0,.35);
            }}
            /* Titles */
            h1, h2, h3 {{
                color: var(--primary) !important;
                letter-spacing: .04em;
                text-shadow: 0 0 12px rgba(0,255,224,.3), 0 0 24px rgba(0,255,224,.15);
            }}
            /* Metrics */
            [data-testid="stMetricValue"] {{
                color: var(--accent) !important;
                text-shadow: 0 0 10px rgba(0,255,136,.5);
            }}
            [data-testid="stMetricLabel"] {{
                color: #9dc6ff !important;
            }}
            /* Tables */
            .stTable td, .stTable th {{
                border-color: #17304a !important;
            }}
            /* Inputs */
            .stSelectbox, .stTextInput, .stNumberInput, .stSlider, .stMultiSelect {{
                filter: drop-shadow(0 0 0 rgba(0,0,0,0));
            }}
            .stSelectbox div[data-baseweb="select"] > div {{
                background: var(--panel) !important;
                border: 1px solid #18324a !important;
            }}
            .stTextInput input, .stNumberInput input {{
                background: var(--panel) !important;
                border: 1px solid #18324a !important;
                color: var(--text) !important;
            }}
            .stMultiSelect div[data-baseweb="select"] > div {{
                background: var(--panel) !important;
                border: 1px solid #18324a !important;
            }}
            .stSlider [data-baseweb="slider"] > div {{
                background: linear-gradient(90deg, rgba(0,255,224,.2), rgba(0,255,136,.2)) !important;
            }}
            /* Buttons */
            .stButton > button {{
                background: linear-gradient(90deg, rgba(0,255,224,.15), rgba(0,255,136,.15));
                border: 1px solid rgba(0,255,224,.35);
                color: var(--primary);
                text-shadow: 0 0 8px rgba(0,255,224,.4);
                border-radius: 12px;
            }}
            .stButton > button:hover {{
                box-shadow: 0 0 14px rgba(0,255,224,.25), inset 0 0 18px rgba(0,255,136,.12);
                transform: translateY(-1px);
            }}
            /* Expander */
            [data-testid="stExpander"] {{
                border: 1px solid #17304a !important;
                background: #0c1122 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="Fibonacci Day Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_neon_theme()

# =========================
# Data + TA helpers
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
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.reset_index()
    if "Datetime" in df.columns: df = df.rename(columns={"Datetime": "Date"})
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

def confluence_from_timeframes(symbol, tfs, lookback=200, tol_pct=0.2):
    rows=[]
    for tf in tfs:
        dfe = fetch(symbol, tf)
        if dfe.empty: continue
        hi,lo,up = recent_swing(dfe, lookback)
        if hi is None or lo is None: continue
        fib = fib_levels(hi[0], lo[0], up)
        for k,v in fib.items(): rows.append({"tf":tf,"level":k,"price":float(v)})
    if not rows: return [],[]
    rows = sorted(rows, key=lambda r: r["price"])
    clusters=[]; group=[]; base=None
    for r in rows:
        if base is None: base=r["price"]; group=[r]
        elif abs(r["price"]-base)/base*100<=tol_pct: group.append(r)
        else: clusters.append(group); base=r["price"]; group=[r]
    if group: clusters.append(group)
    summaries=[{"y0":round(min(x["price"] for x in g),4),
               "y1":round(max(x["price"] for x in g),4),
               "count":len(g),
               "members":g} for g in clusters]
    return rows, summaries

def plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", opacity=0.98, increasing_line_width=1.2, decreasing_line_width=1.2
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor=BG, plot_bgcolor=PLOTBG,
        font=dict(color="#cde7ff", family="Orbitron"),
        height=620, margin=dict(l=20,r=20,t=50,b=30),
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
    extra_tfs = st.multiselect("Confluence TFs", ["1m","5m","15m","1h","1d"], default=["1h","1d"])
    lookback = st.slider("Swing Lookback (candles)", 50, 500, 120, step=10)
    show_vwap = st.checkbox("VWAP", value=True)
    show_rsi  = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD (12,26,9)", value=False)
    alert_level = st.selectbox("Alert on primary Fib", ["Off","23%","38%","50%","61%","79%"], index=0)

# =========================
# Data & TA
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
_, clusters = confluence_from_timeframes(symbol, [t for t in extra_tfs if t != tf], lookback, 0.2)

# =========================
# Main Chart
# =========================
title = f"{symbol} — {tf} — {price:.2f}"
fig = plot_candles(df.tail(300), title)

# Fib lines
for name, level in fib_primary.items():
    fig.add_hline(y=level,
        line=dict(width=1, dash="dot", color=PRIMARY),
        annotation_text=f"{name}: {level:.2f}",
        annotation_font=dict(color=PRIMARY, family="Orbitron"),
        annotation_bgcolor="rgba(0,0,0,0.35)",
        annotation_position="right",
        opacity=0.85
    )

# Confluence shading
for c in clusters:
    fig.add_hrect(y0=c["y0"], y1=c["y1"], line_width=0,
                  fillcolor="rgba(0,255,136,.14)", opacity=0.18)

# Indicators
if show_vwap:
    fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP",
                             line=dict(width=1.6, color=ACCENT)))
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20",
                         line=dict(width=1.2, color="#6fb1ff")))

st.plotly_chart(fig, use_container_width=True)

# Metrics row
c1,c2,c3,c4 = st.columns(4)
c1.metric("Price", f"{price:.2f}")
if show_rsi:  c2.metric("RSI (14)", f"{float(df['RSI'].iloc[-1]):.1f}")
if show_macd:
    c3.metric("MACD",   f"{float(df['MACD'].iloc[-1]):.3f}")
    c4.metric("Signal", f"{float(df['MACD_sig'].iloc[-1]):.3f}")

# Nearest levels
st.subheader("Nearest Primary Fib Levels")
st.table(pd.DataFrame(nearest_levels(price, fib_primary, n=3)))

# Alerts
if alert_level != "Off":
    key = alert_level
    target = fib_primary.get(key)
    if target is not None and len(df) >= 2:
        prev = float(df["Close"].iloc[-2])
        touched  = abs(price - target)/price <= 0.001
        broken_up = price > target and prev <= target
        broken_dn = price < target and prev >= target
        if touched or broken_up or broken_dn:
            st.success(f"🔔 {symbol}: {key} @ {target:.2f} touched/broken on {tf}")
        else:
            st.info(f"Alert armed: {symbol} {key} @ {target:.2f} (refresh to check)")

# Position sizing
st.markdown("### Position Sizing")
pc1, pc2, pc3, pc4 = st.columns(4)
risk_pct = pc1.number_input("Risk per trade (%)", 0.1, 5.0, 1.0, step=0.1)
entry    = pc2.number_input("Entry", value=float(price), step=0.01, format="%.2f")
default_stop = nearest_levels(price, fib_primary, 1)[0]["Price"] if fib_primary else price*0.99
stop     = pc3.number_input("Stop",  value=float(default_stop), step=0.01, format="%.2f")
acct     = pc4.number_input("Account Size ($)", value=10000, step=100)

risk_amt = acct*(risk_pct/100)
per_share = abs(entry - stop)
shares = int(risk_amt/per_share) if per_share>0 else 0
rr = (fib_primary.get("0%", entry) - entry)/(entry - stop) if (entry - stop)!=0 else np.nan

cc = st.columns(3)
cc[0].write(f"**Max Position**: `{shares}` shares")
cc[1].write(f"**Risk Amount**: `${risk_amt:.2f}`")
cc[2].write(f"**Est. R:R to 0%**: `{rr:.2f}`")

with st.expander("Confluence Details"):
    if clusters:
        st.write(pd.DataFrame([{
            "Zone Low":c["y0"], "Zone High":c["y1"], "Count":c["count"],
            "Members":", ".join([f"{m['tf']} {m['level']}" for m in c["members"]])
        } for c in clusters]))
    else:
        st.write("No confluence clusters found for selected timeframes.")
