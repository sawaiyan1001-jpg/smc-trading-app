import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="SMC AI Trading Assistant", page_icon="📈", layout="wide")

# -------------------- DATA --------------------
@st.cache_data(ttl=20, show_spinner=False)
def crypto_ohlcv(symbol: str, limit: int = 500):
    import ccxt
    ex = ccxt.binance({"enableRateLimit": True})
    rows = ex.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
    df = pd.DataFrame(rows, columns=["timestamp","Open","High","Low","Close","Volume"])
    df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("Datetime")[["Open","High","Low","Close","Volume"]].astype(float)

@st.cache_data(ttl=45, show_spinner=False)
def yahoo_ohlcv(ticker: str, period="5d"):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1m", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("डेटा उपलब्ध नहीं है। Yahoo Finance इस symbol के लिए 1-minute डेटा नहीं दे रहा।")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    needed=["Open","High","Low","Close","Volume"]
    for c in needed:
        if c not in df.columns:
            df[c]=0.0
    df.index = pd.to_datetime(df.index, utc=True)
    return df[needed].dropna(subset=["Open","High","Low","Close"])

def get_data(asset, custom_symbol):
    if asset == "Crypto (Binance)":
        return crypto_ohlcv(custom_symbol, 500)
    symbols = {
        "Gold (Yahoo)": "GC=F",
        "Gold Spot (Yahoo)": "XAUUSD=X",
        "Bitcoin (Yahoo)": "BTC-USD",
        "EUR/USD (Yahoo)": "EURUSD=X",
        "USD/JPY (Yahoo)": "JPY=X",
        "S&P 500 (Yahoo)": "^GSPC",
        "NASDAQ 100 (Yahoo)": "^NDX",
    }
    return yahoo_ohlcv(symbols[asset])

# -------------------- SMC LOGIC --------------------
def pivots(df, left=3, right=3):
    h, l = df["High"].values, df["Low"].values
    sh = np.zeros(len(df), dtype=bool)
    sl = np.zeros(len(df), dtype=bool)
    for i in range(left, len(df)-right):
        sh[i] = h[i] == np.max(h[i-left:i+right+1])
        sl[i] = l[i] == np.min(l[i-left:i+right+1])
    return sh, sl

def analyze(df):
    d=df.copy().dropna()
    if len(d)<80:
        return {"status":"INSUFFICIENT DATA","reason":"कम से कम 80 candles चाहिए।","df":d}
    sh, sl = pivots(d)
    d["SwingHigh"] = np.where(sh, d["High"], np.nan)
    d["SwingLow"] = np.where(sl, d["Low"], np.nan)
    d["ATR"] = (d["High"]-d["Low"]).rolling(14).mean()
    d["VolMA"] = d["Volume"].rolling(20).mean()
    last=d.iloc[-1]
    prev=d.iloc[-2]
    prior_high=d["SwingHigh"].dropna()
    prior_low=d["SwingLow"].dropna()
    recent_high=float(prior_high.iloc[-1]) if len(prior_high) else np.nan
    recent_low=float(prior_low.iloc[-1]) if len(prior_low) else np.nan
    close=float(last.Close); high=float(last.High); low=float(last.Low)
    atr=float(last.ATR) if pd.notna(last.ATR) else float((d.High-d.Low).tail(14).mean())
    atr=max(atr, close*0.00005)
    # Structure bias from recent close vs 20/50 EMA and slope
    ema20=d.Close.ewm(span=20).mean()
    ema50=d.Close.ewm(span=50).mean()
    bias="BULLISH" if ema20.iloc[-1]>ema50.iloc[-1] else "BEARISH"
    # Confirmed swing references exclude last 3 candles to reduce repainting
    confirmed=d.iloc[:-3]
    highs=confirmed["SwingHigh"].dropna()
    lows=confirmed["SwingLow"].dropna()
    ref_hi=float(highs.iloc[-1]) if len(highs) else np.nan
    ref_lo=float(lows.iloc[-1]) if len(lows) else np.nan
    sweep_low = bool(pd.notna(ref_lo) and low < ref_lo and close > ref_lo)
    sweep_high = bool(pd.notna(ref_hi) and high > ref_hi and close < ref_hi)
    displacement = abs(close-float(prev.Close)) > 0.8*atr and abs(close-float(last.Open)) > 0.55*atr
    bull_break = bool(pd.notna(ref_hi) and close > ref_hi)
    bear_break = bool(pd.notna(ref_lo) and close < ref_lo)
    # Three-candle FVG, current candle is third candle
    bull_fvg = bool(len(d)>=3 and d.Low.iloc[-1] > d.High.iloc[-3])
    bear_fvg = bool(len(d)>=3 and d.High.iloc[-1] < d.Low.iloc[-3])
    # approximate OB: last opposite candle among previous 8 before displacement
    bull_ob=bear_ob=np.nan
    for j in range(len(d)-2, max(-1,len(d)-10), -1):
        if d.Close.iloc[j] < d.Open.iloc[j] and close > d.High.iloc[j] and displacement:
            bull_ob=float(d.High.iloc[j]); break
    for j in range(len(d)-2, max(-1,len(d)-10), -1):
        if d.Close.iloc[j] > d.Open.iloc[j] and close < d.Low.iloc[j] and displacement:
            bear_ob=float(d.Low.iloc[j]); break
    # Scores are rule-confluence points, NOT win probabilities.
    buy=0; sell=0; buy_reasons=[]; sell_reasons=[]
    if bias=="BULLISH": buy+=10; buy_reasons.append("EMA20 > EMA50 bias")
    else: sell+=10; sell_reasons.append("EMA20 < EMA50 bias")
    if sweep_low: buy+=25; buy_reasons.append("Sell-side liquidity sweep")
    if sweep_high: sell+=25; sell_reasons.append("Buy-side liquidity sweep")
    if bull_break: buy+=15; buy_reasons.append("Bullish structure break / possible MSS")
    if bear_break: sell+=15; sell_reasons.append("Bearish structure break / possible MSS")
    if displacement and close>float(last.Open): buy+=15; buy_reasons.append("Bullish displacement")
    if displacement and close<float(last.Open): sell+=15; sell_reasons.append("Bearish displacement")
    if bull_fvg: buy+=10; buy_reasons.append("Bullish 3-candle FVG")
    if bear_fvg: sell+=10; sell_reasons.append("Bearish 3-candle FVG")
    if pd.notna(bull_ob): buy+=10; buy_reasons.append("Approx. bullish order block")
    if pd.notna(bear_ob): sell+=10; sell_reasons.append("Approx. bearish order block")
    # continuation pullback proxy: aligned bias + recent FVG in direction
    if bias=="BULLISH" and bull_fvg: buy+=10; buy_reasons.append("Bullish continuation confluence")
    if bias=="BEARISH" and bear_fvg: sell+=10; sell_reasons.append("Bearish continuation confluence")
    direction="WAIT"; score=max(buy,sell); reasons=[]
    entry=sl=tp1=tp2=None
    if buy>=65 and buy>sell:
        direction="BUY"; score=buy; reasons=buy_reasons
        entry=close; sl=min(low, ref_lo) - 0.15*atr if pd.notna(ref_lo) else low-0.8*atr
        risk=max(entry-sl, 0.2*atr); sl=entry-risk; tp1=entry+2*risk; tp2=entry+3*risk
    elif sell>=65 and sell>buy:
        direction="SELL"; score=sell; reasons=sell_reasons
        entry=close; sl=max(high, ref_hi) + 0.15*atr if pd.notna(ref_hi) else high+0.8*atr
        risk=max(sl-entry, 0.2*atr); sl=entry+risk; tp1=entry-2*risk; tp2=entry-3*risk
    # classify setup
    setup="No confirmed setup"
    if direction=="BUY":
        setup="Reversal" if sweep_low and bull_break else "Continuation / bullish confluence"
    if direction=="SELL":
        setup="Reversal" if sweep_high and bear_break else "Continuation / bearish confluence"
    return {"status":direction,"score":int(min(score,100)),"bias":bias,"setup":setup,
            "entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"reasons":reasons,
            "sweep_low":sweep_low,"sweep_high":sweep_high,"bull_fvg":bull_fvg,"bear_fvg":bear_fvg,
            "bull_break":bull_break,"bear_break":bear_break,"displacement":displacement,
            "ref_hi":ref_hi,"ref_lo":ref_lo,"df":d}

# -------------------- UI --------------------
st.title("📈 SMC AI Trading Assistant")
st.caption("1-minute rule-based SMC/ICT research prototype • Signals are not financial advice")
with st.sidebar:
    st.header("Market settings")
    asset=st.selectbox("Market",["Crypto (Binance)","Gold (Yahoo)","Gold Spot (Yahoo)","Bitcoin (Yahoo)","EUR/USD (Yahoo)","USD/JPY (Yahoo)","S&P 500 (Yahoo)","NASDAQ 100 (Yahoo)"])
    custom="BTC/USDT"
    if asset=="Crypto (Binance)":
        custom=st.selectbox("Crypto pair",["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT"])
    refresh=st.selectbox("Refresh interval",["Manual","15 seconds","30 seconds","60 seconds"])
    st.info("Analysis timeframe is fixed at 1 minute.")
    st.caption("Data providers may delay, limit or omit symbols.")

if st.button("🔄 Scan market", type="primary"):
    st.cache_data.clear()

try:
    with st.spinner("Fetching 1-minute candles and analyzing SMC structure..."):
        raw=get_data(asset,custom)
        result=analyze(raw)
    d=result["df"]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Signal",result["status"])
    c2.metric("Confluence score",f'{result.get("score",0)}/100')
    c3.metric("HTF proxy bias",result.get("bias","—"))
    c4.metric("Setup type",result.get("setup","—"))
    if result["status"] in ("BUY","SELL"):
        st.success(f'{result["status"]} candidate • Entry is based on latest completed/available 1m candle. Confirm spread, fees and execution before acting.')
        cols=st.columns(4)
        for col,label,key in zip(cols,["Entry reference","Stop Loss","TP1 (1:2)","TP2 (1:3)"],["entry","sl","tp1","tp2"]):
            val=result.get(key)
            col.metric(label, "—" if val is None else f"{val:.5f}")
    else:
        st.warning(f'WAIT — {result.get("reason","Required confirmations are not aligned. Do not force a trade.")}')
    st.subheader("SMC / ICT confirmations")
    checks=[
        ("Sell-side liquidity sweep",result.get("sweep_low")),
        ("Buy-side liquidity sweep",result.get("sweep_high")),
        ("Bullish structure break / MSS proxy",result.get("bull_break")),
        ("Bearish structure break / MSS proxy",result.get("bear_break")),
        ("Bullish FVG",result.get("bull_fvg")),
        ("Bearish FVG",result.get("bear_fvg")),
        ("Displacement",result.get("displacement")),
    ]
    cols=st.columns(2)
    for i,(name,ok) in enumerate(checks):
        cols[i%2].write(("✅ " if ok else "⬜ ")+name)
    st.write("**Reasons contributing to score:**", ", ".join(result.get("reasons",[])) or "No qualifying confluence.")
    fig=go.Figure(data=[go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="1m")])
    if pd.notna(result.get("ref_hi",np.nan)):
        fig.add_hline(y=result["ref_hi"],line_dash="dot",annotation_text="Recent swing high")
    if pd.notna(result.get("ref_lo",np.nan)):
        fig.add_hline(y=result["ref_lo"],line_dash="dot",annotation_text="Recent swing low")
    if result.get("entry") is not None:
        fig.add_hline(y=result["entry"],line_dash="solid",annotation_text="Entry")
        fig.add_hline(y=result["sl"],line_dash="dash",annotation_text="SL")
        fig.add_hline(y=result["tp1"],line_dash="dash",annotation_text="TP1")
    fig.update_layout(height=560,xaxis_rangeslider_visible=False,title="1-minute candlestick chart")
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f"Last candle timestamp: {d.index[-1]} • Last close: {d.Close.iloc[-1]:.5f}")
except Exception as e:
    st.error(f"डेटा या विश्लेषण में समस्या: {e}")
    st.info("Crypto के लिए Binance public market data उपलब्ध होना चाहिए। Yahoo symbols के 1m डेटा पर provider की सीमाएँ लागू होती हैं।")

st.divider()
st.subheader("महत्वपूर्ण सीमाएँ और सुरक्षा")
st.markdown("""
- यह **पहला working prototype** है, प्रमाणित profitable strategy या autonomous trading bot नहीं।
- 1-minute SMC concepts का algorithmic detection अनुमान-आधारित है; वास्तविक discretionary SMC से पूर्ण समानता नहीं।
- Score केवल नियमों की confluence है, जीतने की probability नहीं।
- 1:2 / 1:3 लक्ष्य गणितीय स्तर हैं; spread, slippage, fees और market gaps परिणाम बदल सकते हैं।
- ऐप अभी order place नहीं करता। पहले paper trading और कम-से-कम कई सौ ऐतिहासिक trades का out-of-sample परीक्षण करें।
- Yahoo Finance / Binance data broker execution prices से अलग हो सकता है। लाइव ट्रेड से पहले अपने broker के candles से मिलान करें।
""")
if refresh!="Manual":
    secs={"15 seconds":15,"30 seconds":30,"60 seconds":60}[refresh]
    st.caption(f"Auto-refresh enabled: {secs} seconds. Keep this page open.")
    time.sleep(secs)
    st.rerun()
