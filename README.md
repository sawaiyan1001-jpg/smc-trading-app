# SMC AI Trading Assistant — 1-minute prototype

This is a mobile-friendly Streamlit web app prototype. It scans 1-minute OHLCV candles and applies transparent, rule-based approximations of selected SMC/ICT concepts:
- recent swing highs/lows and liquidity sweeps
- structure breaks (BOS / possible MSS proxy)
- 3-candle FVG
- approximate order-block detection
- EMA trend bias
- displacement proxy
- reversal and continuation confluence
- illustrative entry / stop / 1:2 and 1:3 target references

## Important
This is **not** a 100%-accurate system and does not predict guaranteed reversals. The displayed confluence score is NOT a win probability. SMC labels such as CHOCH, MSS, order blocks and inducement have discretionary definitions; this prototype uses simplified, explicit heuristics. It is for research and paper trading only. It does not place orders.

## Run on a computer
1. Install Python 3.10 or newer.
2. Extract this ZIP.
3. Open a terminal in this folder.
4. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. Open the local URL shown by Streamlit.

## Use from Android
The easiest route is to deploy this project to Streamlit Community Cloud:
1. Create a GitHub account and a new public/private repository.
2. Upload `app.py` and `requirements.txt`.
3. On Streamlit Community Cloud, create an app from that repository and select `app.py`.
4. Open the resulting app URL on Android Chrome and use **Add to Home screen**.

An APK wrapper can be built later, but the web app is the simplest first version. Hosting, data access and internet connectivity are required for live scanning.

## Data sources
- Crypto: Binance public OHLCV via CCXT, 1-minute, common USDT pairs.
- Gold / FX / index proxies: Yahoo Finance 1-minute data via yfinance, subject to availability, delays, market hours and provider limits.
- These may not match the exact feed on your broker. For production, integrate the broker/exchange's official market-data API.

## Accuracy improvement roadmap
1. Validate signals against the exact broker feed and candle close timestamps.
2. Build a historical event-driven backtester with fees, spread and slippage.
3. Define non-repainting pivot confirmation and exact CHOCH/BOS rules.
4. Add session filters, economic-calendar/news avoidance and volatility filters.
5. Run walk-forward and out-of-sample tests; publish win rate, expectancy, max drawdown and sample size.
6. Paper trade before considering any live execution.

Never put broker passwords, API secrets or exchange withdrawal-enabled keys in this prototype.
