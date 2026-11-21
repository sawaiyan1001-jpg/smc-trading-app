import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Professional SMC Trading Assistant", page_icon="📈", layout="wide")

# -------------------- CONFIG --------------------
API_URL = "https://api.twelvedata.com/time_series"
INTERVALS = {"1 minute": "1min", "5 minutes": "5min", "15 minutes": "15min", "1 hour": "1h", "4 hours": "4h"}
INTERVAL_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "1h": 60, "4h": 240}

FOREX = {
    "EUR/USD": "EUR/USD", "GBP/USD": "GBP/USD", "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD", "USD/CAD": "USD/CAD", "USD/CHF": "USD/CHF",
    "NZD/USD": "NZD/USD", "XAU/USD (Gold)": "XAU/USD",
}
CRYPTO = {"BTC/USD": "BTC/USD", "ETH/USD": "ETH/USD", "SOL/USD": "SOL/USD", "XRP/USD": "XRP/USD"}

# -------------------- DATA --------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_twelvedata(symbol: str, interval: str, outputsize: int = 300):
    api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
    if not api_key:
        raise RuntimeError("Streamlit Secrets में TWELVE_DATA_API_KEY नहीं मिला।")
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "timezone": "UTC",
        "order": "ASC",
        "apikey": api_key,
    }
    try:
        response = requests.get(API_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Twelve Data से कनेक्शन की समस्या: {exc}") from exc

    if payload.get("status") == "error" or "values" not in payload:
        message = payload.get("message", "API ने candle data नहीं लौटाया।")
        code = payload.get("code", "")
        raise RuntimeError(f"Twelve Data error {code}: {message}")

    df = pd.DataFrame(payload["values"])
    if df.empty:
        raise RuntimeError("इस symbol/timeframe के लिए डेटा खाली है।")
    df = df.rename(columns={"datetime": "Datetime", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    df = df.sort_values("Datetime").drop_duplicates("Datetime").set_index("Datetime")
    if len(df) < 80:
        raise RuntimeError(f"केवल {len(df)} candles मिलीं; विश्लेषण के लिए कम से कम 80 चाहिए।")

    # Do not use the still-forming latest candle.
    minutes = INTERVAL_MINUTES[interval]
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and now < (df.index[-1] + pd.Timedelta(minutes=minutes)):
        df = df.iloc[:-1]
    if len(df) < 80:
        raise RuntimeError("अधूरी candle हटाने के बाद 80 से कम candles बचीं।")
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

# -------------------- SMC ANALYSIS --------------------
def pivots(df, left=3, right=3):
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    swing_high = np.zeros(len(df), dtype=bool)
    swing_low = np.zeros(len(df), dtype=bool)
    for i in range(left, len(df) - right):
        window_h = highs[i-left:i+right+1]
        window_l = lows[i-left:i+right+1]
        swing_high[i] = highs[i] == np.max(window_h) and np.sum(window_h == highs[i]) == 1
        swing_low[i] = lows[i] == np.min(window_l) and np.sum(window_l == lows[i]) == 1
    return swing_high, swing_low


def analyze_smc(df):
    d = df.copy().dropna()
    if len(d) < 80:
        return {"status": "WAIT", "reason": "कम से कम 80 completed candles चाहिए।", "df": d}

    sh, sl = pivots(d)
    d["SwingHigh"] = np.where(sh, d["High"], np.nan)
    d["SwingLow"] = np.where(sl, d["Low"], np.nan)
    d["ATR"] = (d["High"] - d["Low"]).rolling(14).mean()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()

    # Exclude the last three bars so swing references are confirmed, not repainting.
    confirmed = d.iloc[:-3]
    highs = confirmed["SwingHigh"].dropna()
    lows = confirmed["SwingLow"].dropna()
    ref_hi = float(highs.iloc[-1]) if len(highs) else np.nan
    ref_lo = float(lows.iloc[-1]) if len(lows) else np.nan
    last = d.iloc[-1]
    prev = d.iloc[-2]
    close, high, low, opn = map(float, [last.Close, last.High, last.Low, last.Open])
    atr = float(last.ATR) if pd.notna(last.ATR) else float((d.High-d.Low).tail(14).mean())
    atr = max(atr, abs(close) * 0.00001)

    ema_bull = bool(last.EMA20 > last.EMA50)
    ema_bear = not ema_bull
    # Use the PREVIOUS candle trend to detect a genuine shift on the latest completed candle.
    prior_trend = "BULLISH" if d.EMA20.iloc[-2] > d.EMA50.iloc[-2] else "BEARISH"
    current_bias = "BULLISH" if ema_bull else "BEARISH"
    sweep_low = bool(pd.notna(ref_lo) and low < ref_lo and close > ref_lo)
    sweep_high = bool(pd.notna(ref_hi) and high > ref_hi and close < ref_hi)
    bull_break = bool(pd.notna(ref_hi) and close > ref_hi)
    bear_break = bool(pd.notna(ref_lo) and close < ref_lo)
    displacement = bool(abs(close - float(prev.Close)) >= 0.8 * atr and abs(close-opn) >= 0.55*atr)
    bull_displacement = displacement and close > opn
    bear_displacement = displacement and close < opn
    bull_fvg = bool(len(d) >= 3 and float(d.Low.iloc[-1]) > float(d.High.iloc[-3]))
    bear_fvg = bool(len(d) >= 3 and float(d.High.iloc[-1]) < float(d.Low.iloc[-3]))

    # Reversal: a liquidity sweep followed by a close beyond the opposite confirmed swing.
    bullish_choch = bool(prior_trend == "BEARISH" and bull_break)
    bearish_choch = bool(prior_trend == "BULLISH" and bear_break)
    bullish_mss = bool(bullish_choch and bull_displacement)
    bearish_mss = bool(bearish_choch and bear_displacement)

    bull_ob = bear_ob = np.nan
    for j in range(len(d)-2, max(-1, len(d)-13), -1):
        if float(d.Close.iloc[j]) < float(d.Open.iloc[j]) and close > float(d.High.iloc[j]) and bull_displacement:
            bull_ob = float(d.Low.iloc[j]); break
    for j in range(len(d)-2, max(-1, len(d)-13), -1):
        if float(d.Close.iloc[j]) > float(d.Open.iloc[j]) and close < float(d.Low.iloc[j]) and bear_displacement:
            bear_ob = float(d.High.iloc[j]); break

    buy_score = 0
    sell_score = 0
    buy_reasons, sell_reasons = [], []
    if ema_bull: buy_score += 10; buy_reasons.append("EMA20 ऊपर EMA50")
    else: sell_score += 10; sell_reasons.append("EMA20 नीचे EMA50")
    if sweep_low: buy_score += 20; buy_reasons.append("Sell-side liquidity sweep")
    if sweep_high: sell_score += 20; sell_reasons.append("Buy-side liquidity sweep")
    if bullish_choch: buy_score += 20; buy_reasons.append("Bullish CHOCH / structure shift")
    if bearish_choch: sell_score += 20; sell_reasons.append("Bearish CHOCH / structure shift")
    if bull_displacement: buy_score += 15; buy_reasons.append("Bullish displacement")
    if bear_displacement: sell_score += 15; sell_reasons.append("Bearish displacement")
    if bull_fvg: buy_score += 10; buy_reasons.append("Bullish FVG")
    if bear_fvg: sell_score += 10; sell_reasons.append("Bearish FVG")
    if pd.notna(bull_ob): buy_score += 10; buy_reasons.append("Bullish order-block proxy")
    if pd.notna(bear_ob): sell_score += 10; sell_reasons.append("Bearish order-block proxy")

    # Require reversal sequence to be stronger than a lone EMA signal.
    buy_reversal_ok = bullish_choch and (sweep_low or bull_displacement) and (bull_fvg or pd.notna(bull_ob))
    sell_reversal_ok = bearish_choch and (sweep_high or bear_displacement) and (bear_fvg or pd.notna(bear_ob))
    direction, reasons, score = "WAIT", [], max(buy_score, sell_score)
    entry = stop = tp1 = tp2 = None
    if buy_reversal_ok and buy_score >= 55 and buy_score > sell_score:
        direction, reasons, score = "BUY", buy_reasons, buy_score
        entry = close
        stop = min(low, ref_lo) - 0.15*atr if pd.notna(ref_lo) else low - 0.8*atr
        risk = max(entry-stop, 0.25*atr)
        stop = entry-risk; tp1 = entry+2*risk; tp2 = entry+3*risk
    elif sell_reversal_ok and sell_score >= 55 and sell_score > buy_score:
        direction, reasons, score = "SELL", sell_reasons, sell_score
        entry = close
        stop = max(high, ref_hi) + 0.15*atr if pd.notna(ref_hi) else high + 0.8*atr
        risk = max(stop-entry, 0.25*atr)
        stop = entry+risk; tp1 = entry-2*risk; tp2 = entry-3*risk

    if bullish_mss:
        structure_event = "Bullish MSS / CHOCH"
    elif bearish_mss:
        structure_event = "Bearish MSS / CHOCH"
    elif bullish_choch:
        structure_event = "Potential bullish CHOCH — displacement confirmation pending"
    elif bearish_choch:
        structure_event = "Potential bearish CHOCH — displacement confirmation pending"
    elif sweep_low:
        structure_event = "Sell-side liquidity swept — structure shift pending"
    elif sweep_high:
        structure_event = "Buy-side liquidity swept — structure shift pending"
    else:
        structure_event = "No confirmed structure shift"

    setup = "No confirmed setup"
    if direction == "BUY": setup = "Bullish reversal candidate" if bullish_choch else "Bullish continuation candidate"
    if direction == "SELL": setup = "Bearish reversal candidate" if bearish_choch else "Bearish continuation candidate"
    return {
        "status": direction, "score": int(min(score,100)), "bias": current_bias, "pre_shift_bias": prior_trend,
        "structure_event": structure_event, "setup": setup,
        "entry": entry, "sl": stop, "tp1": tp1, "tp2": tp2,
        "reasons": reasons, "sweep_low": sweep_low, "sweep_high": sweep_high,
        "bull_fvg": bull_fvg, "bear_fvg": bear_fvg, "bull_break": bull_break,
        "bear_break": bear_break, "bullish_choch": bullish_choch,
        "bearish_choch": bearish_choch, "bullish_mss": bullish_mss,
        "bearish_mss": bearish_mss, "displacement": displacement,
        "ref_hi": ref_hi, "ref_lo": ref_lo, "df": d,
    }

# -------------------- MULTI-TIMEFRAME --------------------
def professional_frames():
    return {"Bias": "4h", "Structure": "1h", "Confirmation": "15min", "Setup": "5min", "Entry": "1min"}


def run_analysis(symbol, frames):
    results, errors = {}, {}
    for label, interval in frames.items():
        try:
            frame = fetch_twelvedata(symbol, interval, 300)
            results[label] = analyze_smc(frame)
        except Exception as exc:
            errors[label] = str(exc)
    return results, errors


def combine_results(results, errors):
    required = ["Bias", "Structure", "Confirmation", "Setup", "Entry"]
    if any(k not in results for k in required):
        return {"status": "WAIT", "score": 0, "reason": "सभी आवश्यक timeframe का डेटा उपलब्ध नहीं है: " + "; ".join(f"{k}: {v}" for k,v in errors.items())}
    bias = results["Bias"]["bias"]
    structure = results["Structure"]
    confirm = results["Confirmation"]
    entry = results["Entry"]
    # For a reversal, structure timeframe must show CHOCH in the opposite direction to HTF bias.
    reversal_up = bias == "BEARISH" and structure["bullish_choch"] and confirm["bias"] == "BULLISH"
    reversal_down = bias == "BULLISH" and structure["bearish_choch"] and confirm["bias"] == "BEARISH"
    setup = results["Setup"]
    entry_buy = (entry["bullish_choch"] or (entry["sweep_low"] and entry["bull_fvg"] and entry["displacement"])) and (setup["bias"] == "BULLISH" or setup["bullish_choch"])
    entry_sell = (entry["bearish_choch"] or (entry["sweep_high"] and entry["bear_fvg"] and entry["displacement"])) and (setup["bias"] == "BEARISH" or setup["bearish_choch"])
    continuation_buy = bias == "BULLISH" and structure["bias"] == "BULLISH" and confirm["bias"] == "BULLISH"
    continuation_sell = bias == "BEARISH" and structure["bias"] == "BEARISH" and confirm["bias"] == "BEARISH"
    status, rationale = "WAIT", "Timeframe confirmations अभी एक दिशा में aligned नहीं हैं।"
    if reversal_up and entry_buy and entry["status"] == "BUY":
        status, rationale = "BUY", "HTF bearish context के बाद bullish structure shift और entry timeframe confirmation।"
    elif reversal_down and entry_sell and entry["status"] == "SELL":
        status, rationale = "SELL", "HTF bullish context के बाद bearish structure shift और entry timeframe confirmation।"
    elif continuation_buy and entry["status"] == "BUY":
        status, rationale = "BUY", "Bullish multi-timeframe alignment और entry confirmation।"
    elif continuation_sell and entry["status"] == "SELL":
        status, rationale = "SELL", "Bearish multi-timeframe alignment और entry confirmation।"

    # Entry/SL/TP are sourced from the selected entry timeframe analysis.
    score = int(round(np.mean([results[k]["score"] for k in required])))
    if status == "WAIT": score = min(score, 49)
    return {"status": status, "score": score, "reason": rationale, "entry_result": entry,
            "reversal_up": reversal_up, "reversal_down": reversal_down}

# -------------------- UI --------------------
st.title("📈 Professional SMC Multi-Timeframe Assistant")
st.caption("Rule-based SMC/ICT research tool • Signals are candidates, not financial advice")

with st.sidebar:
    st.header("Market settings")
    market_type = st.radio("Market type", ["Forex", "Crypto"], horizontal=True)
    pairs = FOREX if market_type == "Forex" else CRYPTO
    selected_label = st.selectbox("Trading pair", list(pairs.keys()))
    symbol = pairs[selected_label]
    st.divider()
    st.subheader("Analysis mode")
    mode = st.radio("Choose mode", ["Professional SMC Mode", "Custom Timeframe Mode"])
    if mode == "Professional SMC Mode":
        st.caption("Bias: 4H • Structure: 1H • Confirmation: 15M • Setup: 5M • Entry: 1M")
        frames = professional_frames()
    else:
        bias_name = st.selectbox("Higher timeframe (bias)", ["4 hours", "1 hour"], index=0)
        structure_name = st.selectbox("Structure timeframe", ["1 hour", "15 minutes", "5 minutes"], index=0)
        confirm_name = st.selectbox("Confirmation timeframe", ["15 minutes", "5 minutes", "1 minute"], index=0)
        setup_name = st.selectbox("Setup timeframe", ["15 minutes", "5 minutes", "1 hour"], index=1)
        entry_name = st.selectbox("Entry timeframe", ["1 minute", "5 minutes", "15 minutes"], index=0)
        frames = {"Bias": INTERVALS[bias_name], "Structure": INTERVALS[structure_name],
                  "Confirmation": INTERVALS[confirm_name], "Setup": INTERVALS[setup_name], "Entry": INTERVALS[entry_name]}
    refresh = st.selectbox("Refresh", ["Manual", "60 seconds", "5 minutes"])
    st.caption("API requests are cached for 60 seconds. Multiple timeframes use multiple API calls.")

scan = st.button("🔍 Scan market", type="primary", use_container_width=True)
if scan:
    st.cache_data.clear()

if scan or "smc_last_results" in st.session_state:
    if scan:
        with st.spinner("Twelve Data candles ला रहे हैं और SMC structure जाँच रहे हैं..."):
            results, errors = run_analysis(symbol, frames)
            combined = combine_results(results, errors)
            st.session_state["smc_last_results"] = results
            st.session_state["smc_last_errors"] = errors
            st.session_state["smc_last_combined"] = combined
            st.session_state["smc_last_symbol"] = symbol
            st.session_state["smc_last_frames"] = frames
    results = st.session_state.get("smc_last_results", {})
    errors = st.session_state.get("smc_last_errors", {})
    combined = st.session_state.get("smc_last_combined", {"status":"WAIT", "score":0, "reason":"पहले Scan market दबाएँ।"})

    if st.session_state.get("smc_last_symbol") != symbol or st.session_state.get("smc_last_frames") != frames:
        st.info("आपकी चुनी हुई settings बदल गई हैं। नया विश्लेषण देखने के लिए Scan market दबाएँ।")

    c1, c2, c3 = st.columns(3)
    c1.metric("Multi-timeframe signal", combined["status"])
    c2.metric("Confluence score", f'{combined.get("score",0)}/100')
    c3.metric("Timeframes loaded", f'{len(results)}/5')
    if combined["status"] in ("BUY", "SELL"):
        st.success(f'{combined["status"]} candidate — {combined["reason"]}')
        er = combined.get("entry_result", {})
        cols = st.columns(4)
        for col, label, key in zip(cols, ["Entry reference", "Stop Loss", "TP1 (1:2)", "TP2 (1:3)"], ["entry", "sl", "tp1", "tp2"]):
            value = er.get(key)
            col.metric(label, "—" if value is None else f"{value:.5f}")
    else:
        st.warning("WAIT / NO TRADE — " + combined.get("reason", "Confirmations पूरी नहीं हैं।"))

    st.subheader("Multi-timeframe dashboard")
    rows = []
    for label in ["Bias", "Structure", "Confirmation", "Setup", "Entry"]:
        if label in results:
            r = results[label]
            rows.append({"Role": label, "Timeframe": frames[label], "Trend bias": r.get("bias", "—"),
                         "Structure event": r.get("structure_event", "—"), "Local signal": r.get("status", "WAIT"),
                         "Score": r.get("score", 0), "Latest candle (UTC)": str(r["df"].index[-1]) if len(r.get("df", [])) else "—", "Candles": len(r.get("df", []))})
        else:
            rows.append({"Role": label, "Timeframe": frames[label], "Trend bias": "—", "Structure event": errors.get(label, "No data"), "Local signal": "WAIT", "Score": 0, "Latest candle (UTC)": "—", "Candles": 0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Structure Shift / Reversal monitor")
    for label in ["Bias", "Structure", "Confirmation", "Setup", "Entry"]:
        if label in results:
            r = results[label]
            st.write(f"**{label} ({frames[label]})** — {r.get('structure_event','—')}")
    st.caption("CHOCH/MSS algorithmic proxy है: यह confirmed swing level के पार candle close और displacement पर आधारित है। इसे discretionary ICT/SMC analysis का पूर्ण विकल्प न मानें।")

    if "Entry" in results:
        er = results["Entry"]
        d = er["df"]
        fig = go.Figure(data=[go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name=frames["Entry"])])
        if pd.notna(er.get("ref_hi", np.nan)):
            fig.add_hline(y=er["ref_hi"], line_dash="dot", annotation_text="Confirmed swing high")
        if pd.notna(er.get("ref_lo", np.nan)):
            fig.add_hline(y=er["ref_lo"], line_dash="dot", annotation_text="Confirmed swing low")
        if er.get("entry") is not None:
            for key, label, dash in [("entry", "Entry", "solid"), ("sl", "Stop Loss", "dash"), ("tp1", "TP1", "dash"), ("tp2", "TP2", "dot")]:
                fig.add_hline(y=er[key], line_dash=dash, annotation_text=label)
        fig.update_layout(height=560, xaxis_rangeslider_visible=False, title=f"{symbol} — Entry timeframe chart ({frames['Entry']})")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("SMC confirmations — Entry timeframe")
    if "Entry" in results:
        er = results["Entry"]
        checks = [
            ("Sell-side liquidity sweep", er.get("sweep_low")),
            ("Buy-side liquidity sweep", er.get("sweep_high")),
            ("Bullish CHOCH", er.get("bullish_choch")),
            ("Bearish CHOCH", er.get("bearish_choch")),
            ("Bullish MSS", er.get("bullish_mss")),
            ("Bearish MSS", er.get("bearish_mss")),
            ("Bullish FVG", er.get("bull_fvg")),
            ("Bearish FVG", er.get("bear_fvg")),
            ("Displacement", er.get("displacement")),
        ]
        cols = st.columns(2)
        for i, (name, passed) in enumerate(checks):
            cols[i % 2].write(("✅ " if passed else "⬜ ") + name)
        st.write("**Entry timeframe reasons:** " + (", ".join(er.get("reasons", [])) or "कोई qualifying confluence नहीं।"))

    if errors:
        with st.expander("Data/API warnings"):
            for role, message in errors.items():
                st.warning(f"{role}: {message}")

st.divider()
st.subheader("महत्वपूर्ण सीमाएँ और सुरक्षा")
st.markdown("""
- यह नियम-आधारित SMC research prototype है, प्रमाणित profitable strategy या autonomous trading bot नहीं।
- Confluence score जीतने की probability नहीं है। 80/90 score का अर्थ 80/90% accuracy नहीं है।
- CHOCH, MSS, BOS, FVG और Order Block की पहचान algorithmic approximation है; वास्तविक SMC में context और discretion भी महत्वपूर्ण हैं।
- ऐप अभी कोई order place नहीं करता। पहले demo/paper trading और out-of-sample backtesting करें।
- Entry/SL/TP केवल reference levels हैं; spread, slippage, fees, liquidity और broker pricing परिणाम बदल सकते हैं।
- News calendar filter इस संस्करण में शामिल नहीं है; आर्थिक खबरों के समय अलग से सावधानी रखें।
- Free API plan में symbol coverage, intraday history, rate limits और credits बदल सकते हैं। API error आने पर ऐप signal के बजाय WAIT दिखाएगा।
""")

if refresh != "Manual":
    seconds = 60 if refresh == "60 seconds" else 300
    st.caption(f"Auto-refresh: {refresh}. पेज खुला रखें।")
    time.sleep(seconds)
    st.rerun()
