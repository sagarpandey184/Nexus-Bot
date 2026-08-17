import datetime
import time
import requests
import pandas as pd
import ta
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from google import genai
from google.genai import types

st.set_page_config(page_title="NEXUS SMC Trading Terminal", layout="wide", initial_sidebar_state="expanded")
st_autorefresh(interval=30 * 1000, key="nexus_sync_loop")

# =====================================================================
# NEXUS FULL FRAMEWORK PROMPTS
# =====================================================================
NEXUS_DECISION_PROMPT = """
You are the NEXUS SMC/ICT Trading Analysis Engine. This is strictly an ANALYSIS-ONLY system —
you never instruct a live trade, you only present structured analysis. The trader decides.

=====================================================================
ABSOLUTE RULE — NO GUESSING
=====================================================================
You will be given a DATA COMPLETENESS block at the top of every prompt. If ANY required
component is marked MISSING or INSUFFICIENT, you MUST output only:

    NEXUS SIGNAL SKIPPED — Incomplete data: [list exactly which components are missing]

Do NOT attempt a partial analysis, do NOT guess, do NOT fill gaps from prior candles.
Only proceed to full analysis if the completeness block confirms ALL required components
are present: Daily HTF context, 4H candles, 15m candles, 5m Spot candles, ATM CE 5m candles,
ATM PE 5m candles, and Option Chain snapshot (Max Pain, PCR, OI walls).

=====================================================================
NEXUS CORE RULES (apply only when data is complete)
=====================================================================
1. Two-Layer Multi-Timeframe Analysis (mandatory — do NOT skip to a blended read):

   LAYER A — INDEPENDENT PER-TIMEFRAME ANALYSIS.
   Analyze 4H, 15m, and 5m EACH ON ITS OWN FIRST, as if it were the only chart you had.
   For EACH of the three timeframes, independently determine:
     - Structure: BOS or CHoCH/MSS on that timeframe, at what level
     - Liquidity: Pool / Grab / confirmed Sweep on that timeframe (External, Internal/EQH-EQL,
       or Trend Liquidity), at what level
     - Order Block: type (Mitigation/Breaker/Refined) and zone on that timeframe, tested/untested
     - FVG: Fresh / Partial Fill / Mitigated / Inverse FVG on that timeframe, with the range
     - PO3 Phase: Accumulation / Manipulation / Distribution — judged using ONLY that timeframe's
       own price action, not borrowed from another timeframe
   Do this three times, once per timeframe, before moving to Layer B. Never merge them early.

   LAYER B — ALIGNMENT SYNTHESIS.
   After all three independent reads are done, compare them explicitly:
     - Do the 4H and 15m PO3 phases agree or conflict?
     - Is the 15m sweep occurring at a level the 4H chart also marks as significant liquidity
       (agreement = high-quality), or is it happening somewhere the 4H chart is indifferent to
       (weaker, more likely noise)?
     - Is the 5m MSS/CHoCH occurring in the SAME direction as the 4H trend (continuation,
       higher confidence) or AGAINST it (counter-trend, must be treated as Low confidence
       or NO TRADE unless there is a confirmed 4H-level reversal too)?
     - State the alignment explicitly as: FULL ALIGNMENT / PARTIAL ALIGNMENT / CONFLICT.
   The final DECISION and CONFIDENCE must be derived from this Layer B synthesis, not from
   any single timeframe in isolation. A 5m setup that looks perfect on its own but conflicts
   with 4H structure must be downgraded to Low confidence or NO TRADE, and you must say why.

2. Core Structure: Track BOS (trend continuation) vs CHoCH/MSS (reversal confirmation) on every timeframe.

3. Liquidity: Flag Liquidity Pools, Liquidity Grabs, and confirmed Liquidity Sweeps (grab + close
   back inside structure). Distinguish External Liquidity (major swing highs/lows) vs Internal
   Liquidity (equal highs/lows, EQH/EQL, inside a range) vs Trend Liquidity (pullback highs/lows
   within an established trend).

4. Order Blocks: Identify and label the specific sub-type — Mitigation Block, Breaker Block, or
   Refined Order Block. State tested/untested.

5. Fair Value Gaps (FVG): Track 3-candle imbalances as Fresh, Partial Fill, or Mitigated. Separately
   and explicitly flag if any FVG has flipped polarity (a bullish FVG broken and now acting as
   resistance, or vice versa) — label this specifically as "Inverse FVG" in your per-timeframe
   read, not folded silently into the general FVG line.

6. Power of 3 (PO3): Classify the current phase — Accumulation (no signal), Manipulation
   (fake-out/trap — NEVER suggest an entry here even if momentum looks strong), or
   Distribution (the real move — this is the only phase where a trade signal is valid).

7. Smart Money Trap Check: If price just broke a level and the move "looks too easy" / obviously
   inviting retail entries, flag Manipulation risk explicitly before treating it as Distribution.

8. Fibonacci Extension Targets: You will be given the most recent significant swing points
   (Point 1 = origin, Point 2 = impulse end, Point 3 = retracement). Use the supplied Fib
   levels (1.0, 1.272, 1.618, 2.0, 2.618) to state T1 (partial), T2 (partial), T3 (main target,
   the 1.618 level), and T4 (runner, the 2.0 level). Never invent Fibonacci levels — only use
   the ones provided in the data block.

9. Option Chain / OI Squeeze Verification (cross-check spot structure against real positioning):
   - Bullish Short-Covering Squeeze: spot has 2 consecutive 5m closes above a major Call OI
     strike AND that strike shows negative Change in OI (unwinding) in the supplied chain data.
     Only then is a bullish squeeze confirmed — target the next major Call OI wall.
   - Bearish Long-Unwinding Cascade: spot breaks/fails to reclaim a major Put OI strike within
     15 minutes AND that strike shows negative Change in OI AND fresh Call OI is layering above.
     Only then is a bearish cascade confirmed — target the next major Put OI wall.
   - Do NOT call a squeeze from spot price action alone — it must be confirmed by the OI change
     data supplied. If OI data doesn't confirm it, say so explicitly and downgrade confidence.
   - PCR Extreme Fallacy: do not treat an extreme PCR alone as a reversal signal on a day
     already showing strong 4H trend confirmation — note the extreme PCR but do not let it
     override structure.
   - Expiry Day Check: if today is the expiry date noted in the data block and current time
     is after 13:30 IST, treat any apparent OI unwinding as POSSIBLE ROLLOVER, not a genuine
     squeeze, and state this caveat explicitly.

10. Risk & Extension Checks: Auto-downgrade confidence if RSI is diverging from price, if price
    has already traveled a large distance from its origin move, if the same OB/FVG has been
    retested repeatedly without breaking, or if stacking risk exists (multiple compounding
    risk factors) — especially on expiry days.

11. Divergent Heavyweights: You will be given the current 5m direction of HDFC Bank, Reliance,
    ICICI Bank, and Infosys. If Nifty is breaking a resistance/support level but a majority of
    these heavyweights are moving the opposite direction, flag this explicitly as elevated
    trap risk and downgrade confidence — a Nifty breakout unconfirmed by its own largest
    constituents is a classic OI-trap warning sign. If data for a name is unavailable, note
    it as unavailable rather than assuming agreement.

=====================================================================
OUTPUT FORMAT — you MUST output exactly this schema, nothing more, nothing less
(unless the completeness check fails, in which case output only the SKIPPED line above):
=====================================================================
[Instrument] | [Timeframe] candle closed

--- LAYER A: INDEPENDENT PER-TIMEFRAME READ ---
[4H] Structure: [BOS/CHoCH] at [level] | Liquidity: [Pool/Grab/Sweep, type, level] | OB: [type+zone, tested/untested] | FVG: [range+type] | PO3: [Accumulation/Manipulation/Distribution]
[15m] Structure: [BOS/CHoCH] at [level] | Liquidity: [Pool/Grab/Sweep, type, level] | OB: [type+zone, tested/untested] | FVG: [range+type] | PO3: [Accumulation/Manipulation/Distribution]
[5m] Structure: [BOS/CHoCH/MSS] at [level] | Liquidity: [Pool/Grab/Sweep, type, level] | OB: [type+zone, tested/untested] | FVG: [range+type] | PO3: [Accumulation/Manipulation/Distribution]

--- LAYER B: ALIGNMENT SYNTHESIS ---
Alignment: [FULL ALIGNMENT / PARTIAL ALIGNMENT / CONFLICT]
Reasoning: [1-2 lines on whether 15m sweep matches 4H liquidity, whether 5m MSS agrees with 4H trend direction]

Daily Bias: [Bullish/Bearish/Neutral]
RSI (5m): [value] — [reading]
Option Chain Read: [Max Pain level] | PCR near ATM: [value] | Nearest Call wall: [strike] | Nearest Put wall: [strike]
OI Squeeze Check: [Confirmed Bullish / Confirmed Bearish / Not Confirmed] — [reason]
Fibonacci Targets: T1 [level] | T2 [level] | T3 (main, 1.618) [level] | T4 (runner, 2.0) [level]

→ DECISION: [CALL / PUT / NO TRADE]
→ CONFIDENCE: [High / Moderate / Low chance] — must reference the Layer B alignment explicitly
→ ENTRY TRIGGER: [exact condition]
→ TARGET: [level(s)]
→ STOP-LOSS: [exact level]
→ INVALIDATION: [what flips this thesis]
"""

NEXUS_CHAT_PROMPT = """
You are the NEXUS SMC/ICT Senior Trading Partner in an interactive chat panel.
You have live access to: Daily HTF context, 4H/15m/5m Spot structure, ATM CE & PE 5m premiums
and OI, the full Option Chain snapshot (Max Pain, PCR, OI walls, Change in OI), and the most
recent automated 5-minute decision.

RULES:
- Never instruct a live trade — this is analysis-only. The trader decides.
- If the trader asks a question that requires data you were not given in this session
  (e.g., a timeframe or instrument not currently loaded), say so explicitly and ask for it —
  do not guess or fabricate levels.
- Answer strictly according to the NEXUS Rulebook: sweeps, breakers, mitigations, FVG types,
  PO3 phases, OI squeeze rules, Fibonacci extension logic, stacking risk, and gap-fade patterns.
- Keep answers dense and objective, referencing the actual live levels provided in context.
"""

# =====================================================================
# UPSTOX MULTI-ASSET DATA ENGINE
# =====================================================================
class UpstoxEngine:
    def __init__(self, token):
        self.token = token
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
        self.spot_key = "NSE_INDEX|Nifty 50"
        self.token_valid = None  # None = not yet checked

    def validate_token(self):
        """Hits a lightweight endpoint to confirm the token is accepted. Returns (bool, message)."""
        try:
            res = requests.get("https://api.upstox.com/v2/user/profile", headers=self.headers, timeout=10)
            if res.status_code == 200:
                self.token_valid = True
                name = res.json().get("data", {}).get("user_name", "User")
                return True, f"✅ Upstox token accepted — connected as {name}."
            elif res.status_code == 401:
                self.token_valid = False
                return False, "❌ Upstox token REJECTED or expired. Please generate a new token and re-enter it."
            else:
                self.token_valid = False
                return False, f"❌ Upstox connection error (HTTP {res.status_code}). Check token and try again."
        except Exception as e:
            self.token_valid = False
            return False, f"❌ Could not reach Upstox API: {e}"

    def _is_unauthorized(self, res):
        return res is not None and res.status_code == 401

    def fetch_daily_context(self):
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=45)
        url = f"https://api.upstox.com/v2/historical-candle/{self.spot_key}/day/{today}/{from_date}"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if self._is_unauthorized(res):
                self.token_valid = False
                return None
            if res.status_code != 200:
                return None
            candles = res.json().get("data", {}).get("candles", [])
            if not candles or len(candles) < 14:
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

            # Recent significant swing for Fibonacci reference (last 15 daily candles)
            recent = df.tail(15)
            swing_high = recent["high"].max()
            swing_high_idx = recent["high"].idxmax()
            swing_low = recent["low"].min()
            swing_low_idx = recent["low"].idxmin()

            return {
                "df": df, "pdh": h, "pdl": l, "pdc": c,
                "cpr": {"pivot": round(pivot, 2), "tc": round(tc, 2), "bc": round(bc, 2), "type": cpr_type},
                "daily_rsi": round(daily_rsi, 2),
                "daily_bias": daily_bias,
                "swing_high": round(swing_high, 2),
                "swing_low": round(swing_low, 2),
                "swing_high_after_low": swing_high_idx > swing_low_idx,
            }
        except Exception:
            return None

    def fetch_intraday_candles(self, instrument_key):
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/1minute"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if self._is_unauthorized(res):
                self.token_valid = False
                return pd.DataFrame()
            if res.status_code != 200:
                return pd.DataFrame()
            candles = res.json().get("data", {}).get("candles", [])
            if not candles:
                return pd.DataFrame()

            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
            return df
        except Exception:
            return pd.DataFrame()

    def resample_tf(self, df_1m, rule="5min"):
        """Resamples anchored to today's 9:15 market open so timeframe boundaries are correct."""
        if df_1m.empty:
            return pd.DataFrame()
        today = df_1m.index[0].normalize()
        market_open = today + pd.Timedelta(hours=9, minutes=15)
        resampled = df_1m.resample(rule, origin=market_open).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "oi": "last"
        }).dropna().reset_index()

        if len(resampled) >= 14:
            resampled["rsi"] = ta.momentum.RSIIndicator(resampled["close"], window=14).rsi()
        else:
            resampled["rsi"] = None
        return resampled

    def get_atm_option_keys(self, spot_price):
        if not spot_price or spot_price <= 0:
            return None, None, None, None
        atm_strike = round(spot_price / 50.0) * 50
        url = f"https://api.upstox.com/v2/option/contract?instrument_key={self.spot_key}"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if self._is_unauthorized(res):
                self.token_valid = False
                return None, None, atm_strike, None
            if res.status_code != 200:
                return None, None, atm_strike, None

            contracts = res.json().get("data", [])
            current_date = datetime.date.today().strftime("%Y-%m-%d")
            valid = [c for c in contracts if c.get("expiry", "") >= current_date]
            valid.sort(key=lambda x: x["expiry"])
            nearest_expiry = valid[0]["expiry"] if valid else None

            same_expiry = [c for c in valid if c.get("expiry") == nearest_expiry and c.get("strike_price") == atm_strike]
            ce_key = next((c["instrument_key"] for c in same_expiry if c["instrument_type"] == "CE"), None)
            pe_key = next((c["instrument_key"] for c in same_expiry if c["instrument_type"] == "PE"), None)
            return ce_key, pe_key, atm_strike, nearest_expiry
        except Exception:
            return None, None, atm_strike, None

    def fetch_heavyweights(self):
        """Fetches last two 5m-equivalent closes for top index heavyweights to check for
        divergence against a Nifty breakout/breakdown (Rule 11)."""
        heavyweights = {
            "HDFC Bank": "NSE_EQ|INE040A01034",
            "Reliance": "NSE_EQ|INE002A01018",
            "ICICI Bank": "NSE_EQ|INE090A01021",
            "Infosys": "NSE_EQ|INE009A01021",
        }
        results = {}
        for name, key in heavyweights.items():
            try:
                df_1m = self.fetch_intraday_candles(key)
                if df_1m.empty:
                    results[name] = "No data"
                    continue
                df_5m = self.resample_tf(df_1m, "5min")
                if len(df_5m) < 2:
                    results[name] = "Insufficient data"
                    continue
                last_two = df_5m.tail(2)
                direction = "Rising" if last_two.iloc[-1]["close"] > last_two.iloc[-2]["close"] else "Falling"
                results[name] = f"{direction} ({last_two.iloc[-1]['close']})"
            except Exception:
                results[name] = "Fetch error"
        return results

    def fetch_option_chain(self, expiry):
        """Fetches the FULL option chain for Max Pain, PCR, and OI-wall / Change-in-OI analysis."""
        if not expiry:
            return None
        url = f"https://api.upstox.com/v2/option/chain?instrument_key={self.spot_key}&expiry_date={expiry}"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if self._is_unauthorized(res):
                self.token_valid = False
                return None
            if res.status_code != 200:
                return None
            data = res.json().get("data", [])
            if not data:
                return None

            rows = []
            for row in data:
                strike = row.get("strike_price")
                call = row.get("call_options", {}).get("market_data", {})
                put = row.get("put_options", {}).get("market_data", {})
                call_oi = call.get("oi", 0) or 0
                put_oi = put.get("oi", 0) or 0
                call_oi_chg = call.get("oi_change", call.get("prev_oi_change", 0)) or 0
                put_oi_chg = put.get("oi_change", put.get("prev_oi_change", 0)) or 0
                rows.append({
                    "strike": strike,
                    "call_oi": call_oi, "call_oi_chg": call_oi_chg,
                    "put_oi": put_oi, "put_oi_chg": put_oi,
                    "pcr": round(put_oi / call_oi, 2) if call_oi else None
                })

            chain_df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
            if chain_df.empty:
                return None

            total_call_oi = chain_df["call_oi"].sum()
            total_put_oi = chain_df["put_oi"].sum()
            overall_pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi else None

            # Max Pain: strike with minimum total option-writer payout
            strikes = chain_df["strike"].tolist()
            pain = []
            for s in strikes:
                call_loss = ((chain_df["strike"] < s) * 0).sum()  # placeholder init
                total_loss = 0
                for _, r in chain_df.iterrows():
                    if s > r["strike"]:
                        total_loss += (s - r["strike"]) * r["call_oi"]
                    if s < r["strike"]:
                        total_loss += (r["strike"] - s) * r["put_oi"]
                pain.append((s, total_loss))
            max_pain_strike = min(pain, key=lambda x: x[1])[0] if pain else None

            call_wall = chain_df.loc[chain_df["call_oi"].idxmax(), "strike"] if not chain_df["call_oi"].isna().all() else None
            put_wall = chain_df.loc[chain_df["put_oi"].idxmax(), "strike"] if not chain_df["put_oi"].isna().all() else None

            return {
                "chain_df": chain_df,
                "max_pain": max_pain_strike,
                "overall_pcr": overall_pcr,
                "call_wall": call_wall,
                "put_wall": put_wall,
                "fetched_at": datetime.datetime.now().strftime("%H:%M:%S")
            }
        except Exception:
            return None


# =====================================================================
# FIBONACCI EXTENSION HELPER
# =====================================================================
def compute_fib_extensions(p1, p2, p3, direction="up"):
    """p1 = origin, p2 = impulse end, p3 = retracement pullback point."""
    if p1 is None or p2 is None or p3 is None:
        return None
    leg = abs(p2 - p1)
    levels = {}
    for lvl in [1.0, 1.272, 1.618, 2.0, 2.618]:
        if direction == "up":
            levels[lvl] = round(p3 + leg * lvl, 2)
        else:
            levels[lvl] = round(p3 - leg * lvl, 2)
    return levels


# =====================================================================
# OI SQUEEZE DETECTION
# =====================================================================
def check_oi_squeeze(spot_5m_df, chain_snapshot, prev_chain_snapshot):
    """Cross-checks price action against real Change-in-OI to confirm/deny a squeeze."""
    if spot_5m_df is None or len(spot_5m_df) < 2 or chain_snapshot is None:
        return "Not Confirmed", "Insufficient data for squeeze check"

    last_two = spot_5m_df.tail(2)
    call_wall = chain_snapshot.get("call_wall")
    put_wall = chain_snapshot.get("put_wall")
    chain_df = chain_snapshot.get("chain_df")

    if call_wall and (last_two["close"] > call_wall).all():
        row = chain_df[chain_df["strike"] == call_wall]
        if not row.empty and row.iloc[0]["call_oi_chg"] < 0:
            return "Confirmed Bullish", f"2 closes above Call wall {call_wall} + negative OI change confirms unwinding"
        return "Not Confirmed", f"Price above Call wall {call_wall} but OI not yet unwinding — possible trap"

    if put_wall and (last_two["close"] < put_wall).all():
        row = chain_df[chain_df["strike"] == put_wall]
        if not row.empty and row.iloc[0]["put_oi_chg"] < 0:
            return "Confirmed Bearish", f"2 closes below Put wall {put_wall} + negative OI change confirms unwinding"
        return "Not Confirmed", f"Price below Put wall {put_wall} but OI not yet unwinding — possible trap"

    return "Not Confirmed", "No wall break detected on last 2 candles"


# =====================================================================
# DATA COMPLETENESS GATE — enforces "no guessing"
# =====================================================================
def check_completeness(daily_ctx, df_4h, df_15m, df_5m, df_ce_5m, df_pe_5m, chain_snapshot):
    missing = []
    if not daily_ctx:
        missing.append("Daily HTF context")
    if df_4h is None or df_4h.empty or len(df_4h) < 5:
        missing.append("4H candles (need 5+ bars)")
    if df_15m is None or df_15m.empty or len(df_15m) < 5:
        missing.append("15m candles (need 5+ bars)")
    if df_5m is None or df_5m.empty or len(df_5m) < 14:
        missing.append("5m Spot candles (need 14+ bars for RSI)")
    if df_ce_5m is None or df_ce_5m.empty:
        missing.append("ATM CE 5m candles")
    if df_pe_5m is None or df_pe_5m.empty:
        missing.append("ATM PE 5m candles")
    if not chain_snapshot:
        missing.append("Option Chain snapshot (Max Pain / PCR / OI walls)")
    return missing


# =====================================================================
# STATE INITIALIZATION
# =====================================================================
for key, default in [
    ("bot_active", False), ("decisions", []), ("messages", []),
    ("last_processed_candle", None), ("daily_ctx", None),
    ("token_status_msg", None), ("token_ok", False),
    ("prev_chain_snapshot", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =====================================================================
# SIDEBAR — TOKEN VALIDATION
# =====================================================================
with st.sidebar:
    st.title("⚙️ NEXUS Controls")

    st.subheader("1. API Keys")
    upstox_token = st.text_input("Upstox Access Token", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")

    st.subheader("2. Activation")
    if not st.session_state.bot_active:
        if st.button("🚀 START NEXUS BOT", use_container_width=True, type="primary"):
            if upstox_token and gemini_key:
                temp_engine = UpstoxEngine(upstox_token)
                ok, msg = temp_engine.validate_token()
                st.session_state.token_status_msg = msg
                st.session_state.token_ok = ok
                if ok:
                    st.session_state.bot_active = True
                    st.session_state.gemini_client = genai.Client(api_key=gemini_key)
                    st.session_state.upstox = temp_engine
                    st.session_state.daily_ctx = st.session_state.upstox.fetch_daily_context()
                    st.rerun()
            else:
                st.error("Enter both Upstox Token and Gemini Key.")
    else:
        if st.button("🛑 STOP BOT", use_container_width=True):
            st.session_state.bot_active = False
            st.session_state.daily_ctx = None
            st.rerun()

    if st.session_state.token_status_msg:
        if st.session_state.token_ok:
            st.success(st.session_state.token_status_msg)
        else:
            st.error(st.session_state.token_status_msg)

    if st.session_state.bot_active and st.session_state.daily_ctx:
        st.divider()
        st.subheader("📊 Session Anchors")
        ctx = st.session_state.daily_ctx
        st.metric("Daily Bias", ctx["daily_bias"])
        st.metric("Daily RSI (14)", ctx["daily_rsi"])
        st.caption(f"**CPR Mode:** {ctx['cpr']['type']}")
        st.caption(f"**Pivot Levels:** TC: {ctx['cpr']['tc']} | P: {ctx['cpr']['pivot']} | BC: {ctx['cpr']['bc']}")
        st.caption(f"**PDH:** {ctx['pdh']} | **PDL:** {ctx['pdl']}")
        st.caption(f"**Recent Swing:** High {ctx['swing_high']} / Low {ctx['swing_low']}")

# =====================================================================
# MAIN — NO CHARTS. DATA-ONLY TERMINAL. VIEW CHARTS ON YOUR UPSTOX APP.
# =====================================================================
st.title("⚡ NEXUS SMC Multi-Asset Terminal (Data & Decision Feed)")
st.caption("Charts are intentionally not rendered here — view live candles in your Upstox app. "
           "This terminal handles data aggregation, framework analysis, and decisions only.")

if not st.session_state.bot_active:
    st.info("👋 Enter your credentials in the left sidebar and tap **START NEXUS BOT** to begin.")
else:
    upstox = st.session_state.upstox
    gemini = st.session_state.gemini_client

    # --- Mid-session token re-check: if a previous call flagged 401, halt and ask for new token ---
    if upstox.token_valid is False:
        st.session_state.bot_active = False
        st.session_state.token_status_msg = "❌ Upstox token expired mid-session. Please enter a NEW token and restart."
        st.session_state.token_ok = False
        st.error(st.session_state.token_status_msg)
        st.rerun()

    df_spot_1m = upstox.fetch_intraday_candles("NSE_INDEX|Nifty 50")
    df_spot_5m = upstox.resample_tf(df_spot_1m, "5min")
    df_spot_15m = upstox.resample_tf(df_spot_1m, "15min")
    df_spot_4h = upstox.resample_tf(df_spot_1m, "240min")

    spot_price = df_spot_5m.iloc[-1]["close"] if not df_spot_5m.empty else (
        st.session_state.daily_ctx["pdc"] if st.session_state.daily_ctx else 0)
    ce_key, pe_key, atm_strike, nearest_expiry = upstox.get_atm_option_keys(spot_price)

    df_ce_5m = pd.DataFrame()
    df_pe_5m = pd.DataFrame()
    if ce_key:
        df_ce_1m = upstox.fetch_intraday_candles(ce_key)
        df_ce_5m = upstox.resample_tf(df_ce_1m, "5min")
    if pe_key:
        df_pe_1m = upstox.fetch_intraday_candles(pe_key)
        df_pe_5m = upstox.resample_tf(df_pe_1m, "5min")

    chain_snapshot = upstox.fetch_option_chain(nearest_expiry)
    heavyweight_snapshot = upstox.fetch_heavyweights()

    # Fibonacci reference points from daily swing (fallback for HTF-scale targets)
    d_ctx = st.session_state.daily_ctx or {}
    fib_direction = "up" if d_ctx.get("swing_high_after_low") else "down"
    fib_levels = compute_fib_extensions(
        d_ctx.get("swing_low") if fib_direction == "up" else d_ctx.get("swing_high"),
        d_ctx.get("swing_high") if fib_direction == "up" else d_ctx.get("swing_low"),
        spot_price,
        direction=fib_direction
    )

    # Re-check token status after all calls above (any 401 flips it)
    if upstox.token_valid is False:
        st.session_state.bot_active = False
        st.session_state.token_status_msg = "❌ Upstox token expired mid-session. Please enter a NEW token and restart."
        st.session_state.token_ok = False
        st.error(st.session_state.token_status_msg)
        st.rerun()

    # =================================================================
    # AUTOMATED 5-MINUTE EVALUATION — GATED BY DATA COMPLETENESS
    # =================================================================
    if not df_spot_5m.empty:
        latest_spot = df_spot_5m.iloc[-1]
        candle_ts = str(latest_spot["timestamp"])

        if st.session_state.last_processed_candle != candle_ts:
            st.session_state.last_processed_candle = candle_ts

            missing = check_completeness(
                st.session_state.daily_ctx, df_spot_4h, df_spot_15m, df_spot_5m,
                df_ce_5m, df_pe_5m, chain_snapshot
            )

            if missing:
                st.session_state.decisions.insert(0, {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "text": f"NEXUS SIGNAL SKIPPED — Incomplete data: {', '.join(missing)}"
                })
            else:
                squeeze_result, squeeze_reason = check_oi_squeeze(
                    df_spot_5m, chain_snapshot, st.session_state.prev_chain_snapshot
                )
                st.session_state.prev_chain_snapshot = chain_snapshot

                chain_df = chain_snapshot["chain_df"]
                near_atm = chain_df[(chain_df["strike"] >= atm_strike - 200) & (chain_df["strike"] <= atm_strike + 200)]
                chain_summary = near_atm.to_string(index=False)

                is_expiry_today = (nearest_expiry == datetime.date.today().strftime("%Y-%m-%d"))
                now_time = datetime.datetime.now().strftime("%H:%M")

                ce_summary = df_ce_5m.tail(6)[["timestamp", "open", "high", "low", "close", "volume", "oi"]].to_string(index=False)
                pe_summary = df_pe_5m.tail(6)[["timestamp", "open", "high", "low", "close", "volume", "oi"]].to_string(index=False)
                spot_5m_summary = df_spot_5m.tail(12)[["timestamp", "open", "high", "low", "close", "volume", "rsi"]].to_string(index=False)
                spot_15m_summary = df_spot_15m.tail(8)[["timestamp", "open", "high", "low", "close", "rsi"]].to_string(index=False)
                spot_4h_summary = df_spot_4h.tail(6)[["timestamp", "open", "high", "low", "close", "rsi"]].to_string(index=False)

                prompt = f"""
                DATA COMPLETENESS: ALL COMPONENTS PRESENT — proceed with full analysis.

                --- DAILY HTF CONTEXT ---
                Daily Bias: {d_ctx.get('daily_bias')} | Daily RSI: {d_ctx.get('daily_rsi')}
                CPR: {d_ctx.get('cpr', {}).get('type')} (Pivot: {d_ctx.get('cpr', {}).get('pivot')}, TC: {d_ctx.get('cpr', {}).get('tc')}, BC: {d_ctx.get('cpr', {}).get('bc')})
                PDH: {d_ctx.get('pdh')} | PDL: {d_ctx.get('pdl')}
                Recent Swing (Fib reference): Low {d_ctx.get('swing_low')} / High {d_ctx.get('swing_high')} (direction: {fib_direction})

                --- 4H STRUCTURE (institutional trend / major swing liquidity) ---
                {spot_4h_summary}

                --- 15M STRUCTURE (liquidity sweep check) ---
                {spot_15m_summary}

                --- 5M SPOT PRICE ACTION (execution timeframe) ---
                Latest Bar: Open={latest_spot['open']}, High={latest_spot['high']}, Low={latest_spot['low']}, Close={latest_spot['close']}, RSI={round(latest_spot['rsi'], 2) if pd.notna(latest_spot['rsi']) else 'N/A'}
                {spot_5m_summary}

                --- ATM {atm_strike} CE RECENT 5M BARS & OI ---
                {ce_summary}

                --- ATM {atm_strike} PE RECENT 5M BARS & OI ---
                {pe_summary}

                --- OPTION CHAIN SNAPSHOT (fetched {chain_snapshot['fetched_at']}) ---
                Expiry: {nearest_expiry} | Is Expiry Today: {is_expiry_today} | Current Time: {now_time} IST
                Max Pain: {chain_snapshot['max_pain']}
                Overall PCR: {chain_snapshot['overall_pcr']}
                Nearest Call OI Wall: {chain_snapshot['call_wall']} | Nearest Put OI Wall: {chain_snapshot['put_wall']}
                Strikes near ATM (strike, call_oi, call_oi_chg, put_oi, put_oi_chg, pcr):
                {chain_summary}

                --- PRE-COMPUTED OI SQUEEZE CHECK (already verified against real OI change — use this, don't recompute) ---
                Result: {squeeze_result}
                Reason: {squeeze_reason}

                --- PRE-COMPUTED FIBONACCI EXTENSION LEVELS (from daily swing, direction: {fib_direction}) ---
                {fib_levels}

                --- INDEX HEAVYWEIGHTS (last 5m direction, for divergence check) ---
                {heavyweight_snapshot}
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
                    st.session_state.decisions.insert(0, {
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "text": f"NEXUS SIGNAL SKIPPED — Gemini API error: {e}"
                    })

    # =================================================================
    # LAYOUT — DATA PANELS + DECISION FEED + CHAT (NO CHARTS)
    # =================================================================
    col_data, col_intel = st.columns([1.0, 1.3])

    with col_data:
        st.subheader("📋 Live Data Snapshot")
        st.metric("Nifty 50 Spot", spot_price)
        st.caption(f"ATM Strike: {atm_strike} | Expiry: {nearest_expiry}")

        if chain_snapshot:
            st.markdown("**Option Chain Read**")
            st.write(f"Max Pain: `{chain_snapshot['max_pain']}` | Overall PCR: `{chain_snapshot['overall_pcr']}`")
            st.write(f"Call Wall: `{chain_snapshot['call_wall']}` | Put Wall: `{chain_snapshot['put_wall']}`")
            near_atm = chain_snapshot["chain_df"]
            near_atm = near_atm[(near_atm["strike"] >= atm_strike - 300) & (near_atm["strike"] <= atm_strike + 300)]
            st.dataframe(near_atm, use_container_width=True, height=260)
        else:
            st.warning("Option chain data not yet available.")

        if fib_levels:
            st.markdown("**Fibonacci Extension Reference (Daily swing)**")
            st.json({str(k): v for k, v in fib_levels.items()})

        st.markdown("**Index Heavyweights (divergence check)**")
        st.json(heavyweight_snapshot)

    with col_intel:
        tab_feed, tab_chat = st.tabs(["⏱️ 5-Min Decisions Feed", "💬 Ask NEXUS Partner"])

        with tab_feed:
            st.caption("Auto-evaluates every 5m candle. Skips with an explicit message if any required data is missing — never guesses.")
            if not st.session_state.decisions:
                st.info("Synchronizing data and awaiting next 5-minute candle boundary...")
            for d in st.session_state.decisions[:8]:
                with st.container(border=True):
                    st.markdown(f"**Candle Timestamp: {d['time']} IST**")
                    st.code(d["text"], language="text")

        with tab_chat:
            st.caption("Real-time SMC trading dialogue with full multi-asset, multi-timeframe, and option-chain awareness")
            chat_container = st.container(height=480)

            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if query := st.chat_input("Ask about order blocks, OI squeezes, Fibonacci targets, daily bias..."):
                st.session_state.messages.append({"role": "user", "content": query})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(query)

                latest_sig = st.session_state.decisions[0]["text"] if st.session_state.decisions else "No candle decision recorded yet."

                chat_context = f"""
                --- LIVE MULTI-ASSET SESSION DATA ---
                Spot Price: {spot_price} | ATM Strike: {atm_strike} | Expiry: {nearest_expiry}
                Daily Bias: {d_ctx.get('daily_bias')} | Daily RSI: {d_ctx.get('daily_rsi')} | CPR: {d_ctx.get('cpr', {}).get('type')}
                PDH: {d_ctx.get('pdh')} | PDL: {d_ctx.get('pdl')}
                Max Pain: {chain_snapshot['max_pain'] if chain_snapshot else 'N/A'} | PCR: {chain_snapshot['overall_pcr'] if chain_snapshot else 'N/A'}
                Call Wall: {chain_snapshot['call_wall'] if chain_snapshot else 'N/A'} | Put Wall: {chain_snapshot['put_wall'] if chain_snapshot else 'N/A'}
                Fibonacci Levels: {fib_levels}

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
