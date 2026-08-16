import datetime
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from google import genai
from google.genai import types

st.set_page_config(page_title="NEXUS SMC Trading Terminal", layout="wide", initial_sidebar_state="expanded")
st_autorefresh(interval=30 * 1000, key="nexus_sync_poller")

# =====================================================================
# NEXUS ICT / SMC ENGINE PROMPTS
# =====================================================================
NEXUS_DECISION_PROMPT = """
You are the NEXUS SMC/ICT Trading Analysis Engine. This is strictly an analysis-only system.
You are evaluating live multi-asset intraday market structure for NIFTY 50 Spot, ATM Call (CE), and ATM Put (PE) options.

NEXUS CORE RULES:
1. Core Structure: Track BOS (trend continuation) vs CHoCH / MSS (reversal confirmation).
2. Liquidity: Flag Liquidity Pools, Liquidity Grabs, and Liquidity Sweeps (grab + confirmed close inside). Note Internal vs External Liquidity and Equal Highs/Lows (EQH/EQL).
3. Order Blocks: Identify Mitigation Block, Breaker Block, or Refined OB (tested/untested).
4. Fair Value Gaps (FVG): Track 3-candle gaps as Fresh, Partial Fill, Mitigated, or Inverse FVG.
5. Power of 3 (PO3): Classify market phase: Accumulation, Manipulation (Trap — DO NOT suggest entries), or Distribution (Tradeable move).
6. Cross-Option Verification: Check if Spot displacement is confirmed by ATM CE/PE premium expansion and Open Interest (OI) buildup or if it is a smart money premium trap.
7. Risk & Extension Checks: Auto-downgrade confidence if RSI is diverging, price is overextended, or stacking risk exists on expiry days.
8. Output Format: You MUST output ONLY the exact 8-line schema below:

[Instrument] | [Timeframe] candle closed
Structure: [Uptrend/Downtrend/Ranging] (BOS/CHoCH at [level])
Liquidity: [Sweep/Grab/None] at [level]
Order Block: [type + zone] (tested/untested)
FVG: [range + type] (filled/unfilled)
Power of 3 Phase: [Accumulation/Manipulation/Distribution]
Daily Bias: [Bullish/Bearish/Neutral]
RSI: [value] — [reading]
→ Suggested bias: [LONG/SHORT/NO TRADE] — your call
"""

NEXUS_CHAT_PROMPT = """
You are the NEXUS SMC/ICT Senior Trading Partner.
You have real-time access to the user's active session: Spot CPR levels, Daily Bias, 5m Spot Candles, ATM CE & PE Premiums, and Multi-Timeframe structures.
Answer questions strictly according to the NEXUS Rulebook (Sweeps, Breakers, Mitigations, FVGs, PO3 phases, Stacking Risk, Gap Fades).
Never instruct live trades; provide dense, objective institutional analysis referencing live levels.
"""

# =====================================================================
# UPSTOX MULTI-ASSET DATA ENGINE
# =====================================================================
class UpstoxEngine:
    def __init__(self, token):
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        self.spot_key = "NSE_INDEX|Nifty 50"

    def fetch_daily_context(self):
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=30)
        url = f"https://api.upstox.com/v2/historical-candle/{self.spot_key}/day/{today}/{from_date}"
        res = requests.get(url, headers=self.headers)
        if res.status_code != 200:
            return None
        
        candles = res.json().get("data", {}).get("candles", [])
        if not candles:
            return None
        
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        h, l, c = prev["high"], prev["low"], prev["close"]
        pivot = (h + l + c) / 3.0
        bc = (h + l) / 2.0
        tc = (pivot - bc) + pivot
        cpr_type = "Narrow CPR (Trending Expected)" if abs(tc - bc) < (pivot * 0.0015) else "Wide CPR (Range Expected)"
        
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        daily_rsi = df["rsi"].iloc[-1]
        daily_bias = "Bullish" if daily_rsi > 55 else ("Bearish" if daily_rsi < 45 else "Neutral")
        
        return {
            "df": df, "pdh": h, "pdl": l, "pdc": c,
            "cpr": {"pivot": round(pivot, 2), "tc": round(tc, 2), "bc": round(bc, 2), "type": cpr_type},
            "daily_rsi": round(daily_rsi, 2),
            "daily_bias": daily_bias
        }

    def fetch_intraday_candles(self, instrument_key):
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/1minute"
        res = requests.get(url, headers=self.headers)
        if res.status_code != 200:
            return pd.DataFrame()
            
        candles = res.json().get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()
            
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
        return df

    def resample_tf(self, df_1m, rule="5min"):
        if df_1m.empty:
            return pd.DataFrame()
        resampled = df_1m.resample(rule, offset="15min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "oi": "last"
        }).dropna().reset_index()
        
        if len(resampled) >= 14:
            resampled["rsi"] = ta.momentum.RSIIndicator(resampled["close"], window=14).rsi()
        else:
            resampled["rsi"] = 50.0
        return resampled

    def get_atm_option_keys(self, spot_price):
        atm_strike = round(spot_price / 50.0) * 50
        url = f"https://api.upstox.com/v2/option/contract?instrument_key={self.spot_key}"
        res = requests.get(url, headers=self.headers)
        if res.status_code != 200:
            return None, None, atm_strike
            
        contracts = res.json().get("data", [])
        current_date = datetime.date.today().strftime("%Y-%m-%d")
        valid = [c for c in contracts if c.get("expiry", "") >= current_date and c.get("strike_price") == atm_strike]
        valid.sort(key=lambda x: x["expiry"])
        
        ce_key = next((c["instrument_key"] for c in valid if c["instrument_type"] == "CE"), None)
        pe_key = next((c["instrument_key"] for c in valid if c["instrument_type"] == "PE"), None)
        return ce_key, pe_key, atm_strike

# =====================================================================
# CHART RENDERING HELPER (PLOTLY)
# =====================================================================
def plot_candlestick_chart(df, title, show_oi=False):
    if df.empty:
        return go.Figure()
    
    rows = 2 if show_oi else 1
    row_heights = [0.7, 0.3] if show_oi else [1.0]
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)
    
    # Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#089981", decreasing_line_color="#F23645"
    ), row=1, col=1)
    
    # Optional OI / Volume Subplot
    if show_oi and "oi" in df.columns:
        fig.add_trace(go.Bar(
            x=df["timestamp"], y=df["oi"], name="Open Interest", marker_color="#2962FF"
        ), row=2, col=1)
        
    fig.update_layout(
        title=title, template="plotly_dark", height=420,
        margin=dict(l=10, r=10, t=35, b=10), xaxis_rangeslider_visible=False
    )
    return fig

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if "bot_active" not in st.session_state:
    st.session_state.bot_active = False
if "decisions" not in st.session_state:
    st.session_state.decisions = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_processed_candle" not in st.session_state:
    st.session_state.last_processed_candle = None
if "daily_ctx" not in st.session_state:
    st.session_state.daily_ctx = None

# =====================================================================
# SIDEBAR CONTROLS & AUTHENTICATION
# =====================================================================
with st.sidebar:
    st.title("⚙️ NEXUS Control Hub")
    
    st.subheader("1. API Authentication")
    upstox_token = st.text_input("Upstox Access Token", type="password", help="Daily token from Upstox Developer Portal")
    gemini_key = st.text_input("Gemini API Key", type="password", help="API key from Google AI Studio")
    
    st.subheader("2. Engine Activation")
    if not st.session_state.bot_active:
        if st.button("🚀 START NEXUS BOT", use_container_width=True, type="primary"):
            if upstox_token and gemini_key:
                st.session_state.bot_active = True
                st.session_state.gemini_client = genai.Client(api_key=gemini_key)
                st.session_state.upstox = UpstoxEngine(upstox_token)
                st.session_state.daily_ctx = st.session_state.upstox.fetch_daily_context()
                st.rerun()
            else:
                st.error("Please enter both Upstox Token and Gemini Key.")
    else:
        if st.button("🛑 STOP BOT", use_container_width=True):
            st.session_state.bot_active = False
            st.session_state.daily_ctx = None
            st.rerun()

    if st.session_state.bot_active and st.session_state.daily_ctx:
        st.divider()
        st.subheader("📊 Session Macro Anchor")
        ctx = st.session_state.daily_ctx
        st.metric("Daily Bias", ctx["daily_bias"])
        st.metric("Daily RSI (14)", ctx["daily_rsi"])
        st.caption(f"**CPR Mode:** {ctx['cpr']['type']}")
        st.caption(f"**CPR TC / P / BC:** {ctx['cpr']['tc']} | {ctx['cpr']['pivot']} | {ctx['cpr']['bc']}")
        st.caption(f"**PDH:** {ctx['pdh']} | **PDL:** {ctx['pdl']}")

# =====================================================================
# MAIN DASHBOARD INTERFACE
# =====================================================================
st.title("⚡ NEXUS SMC Multi-Asset Terminal")

if not st.session_state.bot_active:
    st.info("👋 **Welcome to NEXUS SMC Terminal.** Paste your Upstox Daily Token and Gemini API Key in the left sidebar, then click **'START NEXUS BOT'** to stream live charts and automated institutional decisions.")
else:
    upstox = st.session_state.upstox
    gemini = st.session_state.gemini_client
    
    # 1. Ingest Multi-Asset Data
    df_spot_1m = upstox.fetch_intraday_candles("NSE_INDEX|Nifty 50")
    df_spot_5m = upstox.resample_tf(df_spot_1m, "5min")
    df_spot_15m = upstox.resample_tf(df_spot_1m, "15min")
    df_spot_1h = upstox.resample_tf(df_spot_1m, "60min")
    df_spot_4h = upstox.resample_tf(df_spot_1m, "240min")
    
    # 2. Dynamic ATM Discovery & Option Candles
    spot_price = df_spot_5m.iloc[-1]["close"] if not df_spot_5m.empty else 0
    ce_key, pe_key, atm_strike = upstox.get_atm_option_keys(spot_price)
    
    df_ce_5m = pd.DataFrame()
    df_pe_5m = pd.DataFrame()
    if ce_key:
        df_ce_1m = upstox.fetch_intraday_candles(ce_key)
        df_ce_5m = upstox.resample_tf(df_ce_1m, "5min")
    if pe_key:
        df_pe_1m = upstox.fetch_intraday_candles(pe_key)
        df_pe_5m = upstox.resample_tf(df_pe_1m, "5min")

    # 3. Automated 5-Minute NEXUS AI Decision Engine
    if not df_spot_5m.empty:
        latest_spot = df_spot_5m.iloc[-1]
        candle_ts = str(latest_spot["timestamp"])
        
        if st.session_state.last_processed_candle != candle_ts:
            st.session_state.last_processed_candle = candle_ts
            d_ctx = st.session_state.daily_ctx or {}
            
            ce_summary = df_ce_5m.tail(5)[["timestamp", "open", "high", "low", "close", "volume", "oi"]].to_string() if not df_ce_5m.empty else "N/A"
            pe_summary = df_pe_5m.tail(5)[["timestamp", "open", "high", "low", "close", "volume", "oi"]].to_string() if not df_pe_5m.empty else "N/A"
            spot_summary = df_spot_5m.tail(12)[["timestamp", "open", "high", "low", "close", "volume", "rsi"]].to_string()
            
            prompt = f"""
            Analyze the closed 5-minute candle for NIFTY 50 with ATM Option Premiums.
            
            --- DAILY HTF CONTEXT ---
            Daily Bias: {d_ctx.get('daily_bias')} | Daily RSI: {d_ctx.get('daily_rsi')}
            CPR: {d_ctx.get('cpr', {}).get('type')} (Pivot: {d_ctx.get('cpr', {}).get('pivot')})
            PDH: {d_ctx.get('pdh')} | PDL: {d_ctx.get('pdl')}
            
            --- SPOT 5M PRICE ACTION ---
            Latest Bar: Open={latest_spot['open']}, High={latest_spot['high']}, Low={latest_spot['low']}, Close={latest_spot['close']}, RSI={round(latest_spot['rsi'], 2)}
            Recent 12 Bars:
            {spot_summary}
            
            --- ATM {atm_strike} CE RECENT 5M BARS & OI ---
            {ce_summary}
            
            --- ATM {atm_strike} PE RECENT 5M BARS & OI ---
            {pe_summary}
            """
            
            try:
                res = gemini.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=NEXUS_DECISION_PROMPT, temperature=0.1)
                )
                st.session_state.decisions.insert(0, {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "text": res.text.strip()
                })
            except Exception as e:
                st.sidebar.error(f"Analysis Generation Error: {e}")

    # =====================================================================
    # LAYOUT: MULTI-ASSET CHARTS (LEFT) & INTELLIGENCE CONSOLE (RIGHT)
    # =====================================================================
    col_charts, col_intel = st.columns([1.5, 1.0])

    with col_charts:
        chart_tab1, chart_tab2 = st.tabs(["🔥 5m Primary (Spot + ATM CE/PE)", "🌐 Multi-Timeframe Alignment (15m, 1h, 4h, 1D)"])
        
        with chart_tab1:
            st.plotly_chart(plot_candlestick_chart(df_spot_5m, f"NIFTY 50 Spot — 5m Chart (LTP: {spot_price})"), use_container_width=True)
            
            col_ce, col_pe = st.columns(2)
            with col_ce:
                st.plotly_chart(plot_candlestick_chart(df_ce_5m, f"ATM {atm_strike} CE — 5m Chart (with OI)", show_oi=True), use_container_width=True)
            with col_pe:
                st.plotly_chart(plot_candlestick_chart(df_pe_5m, f"ATM {atm_strike} PE — 5m Chart (with OI)", show_oi=True), use_container_width=True)

        with chart_tab2:
            st.plotly_chart(plot_candlestick_chart(df_spot_15m, "NIFTY 50 — 15-Minute Chart (Structural Swings)"), use_container_width=True)
            st.plotly_chart(plot_candlestick_chart(df_spot_1h, "NIFTY 50 — 1-Hour Chart (Protected Levels)"), use_container_width=True)
            st.plotly_chart(plot_candlestick_chart(df_spot_4h, "NIFTY 50 — 4-Hour Chart (Institutional Trend)"), use_container_width=True)
            if st.session_state.daily_ctx:
                st.plotly_chart(plot_candlestick_chart(st.session_state.daily_ctx["df"], "NIFTY 50 — 1-Day Chart (Macro Structure)"), use_container_width=True)

    with col_intel:
        tab_feed, tab_chat = st.tabs(["⏱️ 5-Min Decisions Feed", "💬 Ask NEXUS Partner"])

        with tab_feed:
            st.caption("Auto-evaluates every 5m candle across Spot, CE, PE, and HTF rules[span_1](start_span)[span_1](end_span)")
            if not st.session_state.decisions:
                st.info("Synchronizing data and awaiting next 5-minute candle boundary...")
            for d in st.session_state.decisions[:8]:
                with st.container(border=True):
                    st.markdown(f"**Candle Timestamp: {d['time']} IST**")
                    st.code(d["text"], language="text")

        with tab_chat:
            st.caption("Real-time SMC trading dialogue with full multi-asset awareness")
            chat_container = st.container(height=480)
            
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if query := st.chat_input("Ask about order blocks, option premium sweeps, or daily bias..."):
                st.session_state.messages.append({"role": "user", "content": query})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(query)

                latest_sig = st.session_state.decisions[0]["text"] if st.session_state.decisions else "No candle decision recorded yet."
                d_ctx = st.session_state.daily_ctx or {}
                
                chat_context = f"""
                --- LIVE MULTI-ASSET SESSION DATA ---
                Spot Price: {spot_price} | ATM Strike: {atm_strike}
                Daily Bias: {d_ctx.get('daily_bias')} | Daily RSI: {d_ctx.get('daily_rsi')} | CPR: {d_ctx.get('cpr', {}).get('type')}
                PDH: {d_ctx.get('pdh')} | PDL: {d_ctx.get('pdl')}
                Latest 5m Signal:
                {latest_sig}

                Trader Query: {query}
                """

                try:
                    chat_res = gemini.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=chat_context,
                        config=types.GenerateContentConfig(system_instruction=NEXUS_CHAT_PROMPT, temperature=0.3)
                    )
                    reply = chat_res.text
                except Exception as e:
                    reply = f"Error consulting Gemini: {e}"

                st.session_state.messages.append({"role": "assistant", "content": reply})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(reply)
