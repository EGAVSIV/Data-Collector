import os
import json
import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

# --- Configuration ---
TF_FOLDERS = {
    '15': 'stockdata_15',
    '1H': 'stockdata_1H',
    'D': 'stockdata_D',
    'W': 'stockdata_W',
    'M': 'stockdata_M'
}

TF_PAIRS = [
    ('15', '1H', '15m -> 1h'),
    ('1H', 'D', '1h -> Daily'),
    ('D', 'W', 'Daily -> Weekly'),
    ('W', 'M', 'Weekly -> Monthly')
]

# Order used for "Timeframe Pair" sorting everywhere in the dashboard
PAIR_ORDER = {label: i for i, (_, _, label) in enumerate(TF_PAIRS)}

# Divergence lookback window & split point (older half vs recent half)
DIVERGENCE_LOOKBACK = 30
DIVERGENCE_SPLIT = 15

# Number of trailing daily sessions used by the "MACD 360 FNO" tab
MACD360_DAYS = 15

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

# Market Bias Reference Dictionary for Tab 1 guide
MARKET_BIAS_GUIDE = [
    {"Market Bias": "Strong Bullish", "Interpretation": "Strong Momentum Continuation: Full bullish alignment. HTF and LTF are accelerating in uptrend territory. High-probability trend-following entry."},
    {"Market Bias": "Bullish Reversal", "Interpretation": "Early Momentum Re-Ignition: HTF is strong above zero; LTF crosses positive from deep oversold. Excellent low-risk buy setup on a bottoming LTF bar."},
    {"Market Bias": "Bullish Pullback", "Interpretation": "Healthy Dip / Profit-Taking: HTF remains strong above zero while LTF experiences a minor pullback. Look for LTF reversal signals to buy the dip."},
    {"Market Bias": "Deep Pullback", "Interpretation": "Intermediate Support Test: LTF drops below zero into correction mode while HTF stays bullish. Wait for LTF to stabilize before taking long positions."},
    {"Market Bias": "Dip Buy / Reversal", "Interpretation": "HTF Bottoming Buy: HTF is turning positive below zero (macro bottom); LTF fires an early bullish entry. Great risk-to-reward long setup."},
    {"Market Bias": "Oversold Accumulation", "Interpretation": "Dual Reversal Setup: Both timeframes crossing positive below zero. Indicates long-term base building; ideal for long-term swing positioning."},
    {"Market Bias": "Neutral / Consolidating", "Interpretation": "Mixed Recovery: HTF trying to turn up, but LTF is fading above zero. Expect choppy, sideways consolidation. Avoid aggressive trades."},
    {"Market Bias": "Complex Reversal Failure", "Interpretation": "Failing Base: HTF recovery attempt losing steam as LTF breaks negative under zero. Stand aside or trade tight ranges."},
    {"Market Bias": "Counter-Trend Buy", "Interpretation": "HTF Pullback Re-Entry: HTF taking profits above zero; LTF turns positive. Offers a quick scalp or swing buy in line with broader trend support."},
    {"Market Bias": "Early Recovery Attempt", "Interpretation": "Consolidation Bounce: HTF in upper-level cooling phase while LTF attempts a low-level bounce. Moderate probability setup; trade with smaller size."},
    {"Market Bias": "Correction in Progress", "Interpretation": "Upper-Level Pullback: Both HTF and LTF taking profits above zero. Price is actively retracing; do not buy until LTF turns positive."},
    {"Market Bias": "Accelerating Correction", "Interpretation": "Deeper Retracement: HTF negative above zero; LTF breaks down under zero. Support levels being tested; wait for LTF reversal."},
    {"Market Bias": "Aggressive Counter-Trend", "Interpretation": "Relief Rally Scalp: HTF in strong downtrend; LTF triggers a short-term buy. High-risk bounce trade; keep profit targets tight."},
    {"Market Bias": "Weak Oversold Bounce", "Interpretation": "Failing Bottom: HTF strongly bearish; LTF attempts a low-level bounce. High failure rate for longs; watch out for bear traps."},
    {"Market Bias": "Bearish Continuation", "Interpretation": "Bear Market Rally Fading: LTF turns negative above zero inside an active HTF downtrend. High-probability Short Entry / Sell Signal."},
    {"Market Bias": "Strong Bearish", "Interpretation": "Maximum Downward Alignment: HTF and LTF accelerating down below zero. Strong bearish momentum. Avoid long trades; stay short or in cash."}
]

# Divergence Type Reference Dictionary for Tab 3 guide
DIVERGENCE_GUIDE = [
    {"Divergence Type": "Regular Bullish Divergence", "Interpretation": "Price makes a lower low while MACD makes a higher low. Downside momentum is fading \u2014 classic reversal warning at the end of a downtrend."},
    {"Divergence Type": "Regular Bearish Divergence", "Interpretation": "Price makes a higher high while MACD makes a lower high. Upside momentum is fading \u2014 classic reversal warning at the end of an uptrend."},
    {"Divergence Type": "Hidden Bullish Divergence", "Interpretation": "Price makes a higher low while MACD makes a lower low. Confirms trend continuation \u2014 a healthy pullback inside an existing uptrend rather than a reversal."},
    {"Divergence Type": "Hidden Bearish Divergence", "Interpretation": "Price makes a lower high while MACD makes a higher high. Confirms trend continuation \u2014 a healthy relief bounce inside an existing downtrend rather than a reversal."}
]

# Recommendation guide shown in the Divergence tab (Point 2)
RECOMMENDATION_GUIDE = [
    {"Recommendation": "STRONG BUY", "Bucket": "Regular Bullish (HTF) + Regular Bullish (LTF)", "Interpretation": "Both timeframes show a classic reversal (Regular) bullish divergence at the same time \u2014 the strongest possible reversal confluence."},
    {"Recommendation": "BUY", "Bucket": "Hidden Bullish (HTF) + Regular Bullish (LTF)", "Interpretation": "HTF is in continuation mode (Hidden) while LTF fires a fresh reversal (Regular) divergence \u2014 a trend-continuation entry trigger."},
    {"Recommendation": "SELL", "Bucket": "Hidden Bearish (HTF) + Regular Bearish (LTF)", "Interpretation": "HTF downtrend continuation (Hidden) with an LTF Regular bearish divergence \u2014 a short-entry trigger inside the larger downtrend."},
    {"Recommendation": "STRONG SELL", "Bucket": "Regular Bearish (HTF) + Regular Bearish (LTF)", "Interpretation": "Both timeframes show a classic reversal (Regular) bearish divergence at the same time \u2014 strong distribution / top signal."},
    {"Recommendation": "WATCH", "Bucket": "Any other combination", "Interpretation": "Divergence present but the HTF/LTF combination does not match a defined high-confidence setup. Treat as informational only."}
]

# --- Technical Indicator Functions ---
def calculate_indicators(df, fast=12, slow=26, signal=9, rsi_period=14):
    if df.empty or len(df) < slow + signal:
        return None

    date_col = 'datetime' if 'datetime' in df.columns else ('date' if 'date' in df.columns else None)
    if date_col:
        df = df.sort_values(date_col)

    close = df['close']

    # 1. MACD Calculation
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # 2. RSI (14) Calculation
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    df['macd'] = macd_line
    df['signal'] = signal_line
    df['rsi'] = rsi
    return df

def get_macd_state(macd_val, signal_val):
    if pd.isna(macd_val) or pd.isna(signal_val):
        return None, "N/A"

    is_pco = macd_val > signal_val
    is_above_zero = macd_val > 0

    if is_pco and is_above_zero:
        return 1, "PCO > 0"
    elif is_pco and not is_above_zero:
        return 2, "PCO < 0"
    elif not is_pco and is_above_zero:
        return 3, "NC > 0"
    else:
        return 4, "NC < 0"

def map_transition(ltf_prev, ltf_curr, htf_curr):
    if None in (ltf_prev, ltf_curr, htf_curr):
        return None

    # 16 Condition Mapping Logic
    if htf_curr == 1:  # HTF: PCO > 0
        if ltf_prev == 3 and ltf_curr == 1: return "Strong Bullish"
        if ltf_prev == 4 and ltf_curr == 2: return "Bullish Reversal"
        if ltf_prev == 1 and ltf_curr == 3: return "Bullish Pullback"
        if ltf_prev == 2 and ltf_curr == 4: return "Deep Pullback"

    elif htf_curr == 2:  # HTF: PCO < 0
        if ltf_prev == 3 and ltf_curr == 1: return "Dip Buy / Reversal"
        if ltf_prev == 4 and ltf_curr == 2: return "Oversold Accumulation"
        if ltf_prev == 1 and ltf_curr == 3: return "Neutral / Consolidating"
        if ltf_prev == 2 and ltf_curr == 4: return "Complex Reversal Failure"

    elif htf_curr == 3:  # HTF: NC > 0
        if ltf_prev == 3 and ltf_curr == 1: return "Counter-Trend Buy"
        if ltf_prev == 4 and ltf_curr == 2: return "Early Recovery Attempt"
        if ltf_prev == 1 and ltf_curr == 3: return "Correction in Progress"
        if ltf_prev == 2 and ltf_curr == 4: return "Accelerating Correction"

    elif htf_curr == 4:  # HTF: NC < 0
        if ltf_prev == 3 and ltf_curr == 1: return "Aggressive Counter-Trend"
        if ltf_prev == 4 and ltf_curr == 2: return "Weak Oversold Bounce"
        if ltf_prev == 1 and ltf_curr == 3: return "Bearish Continuation"
        if ltf_prev == 2 and ltf_curr == 4: return "Strong Bearish"

    return None

def detect_divergence(df, lookback=DIVERGENCE_LOOKBACK, split=DIVERGENCE_SPLIT):
    """
    Classifies MACD-vs-price divergence into one of 4 types by comparing the
    older half of the lookback window against the most recent half:

      - Regular Bullish Divergence : price lower low   + MACD higher low   (reversal, end of downtrend)
      - Hidden Bullish Divergence  : price higher low  + MACD lower low    (continuation, inside uptrend)
      - Regular Bearish Divergence : price higher high + MACD lower high  (reversal, end of uptrend)
      - Hidden Bearish Divergence  : price lower high  + MACD higher high (continuation, inside downtrend)

    Returns None if there isn't enough data or no condition is met.
    """
    if df is None or 'macd' not in df.columns or len(df) < lookback:
        return None

    macd = df['macd']
    older_slice = slice(-lookback, -split)
    recent_slice = slice(-split, None)

    price_low1 = df["low"].iloc[older_slice].min()
    price_low2 = df["low"].iloc[recent_slice].min()
    macd_low1 = macd.iloc[older_slice].min()
    macd_low2 = macd.iloc[recent_slice].min()

    price_high1 = df["high"].iloc[older_slice].max()
    price_high2 = df["high"].iloc[recent_slice].max()
    macd_high1 = macd.iloc[older_slice].max()
    macd_high2 = macd.iloc[recent_slice].max()

    if any(pd.isna(v) for v in [price_low1, price_low2, macd_low1, macd_low2,
                                 price_high1, price_high2, macd_high1, macd_high2]):
        return None

    # Bullish checks (based on swing lows) take priority
    if price_low2 < price_low1 and macd_low2 > macd_low1:
        return "Regular Bullish Divergence"
    if price_low2 > price_low1 and macd_low2 < macd_low1:
        return "Hidden Bullish Divergence"

    # Bearish checks (based on swing highs)
    if price_high2 > price_high1 and macd_high2 < macd_high1:
        return "Regular Bearish Divergence"
    if price_high2 < price_high1 and macd_high2 > macd_high1:
        return "Hidden Bearish Divergence"

    return None


# --- Divergence -> Recommendation mapping ---
def get_divergence_recommendation(symbol, htf_pair_label, ltf_pair_label, htf_div, ltf_div):
    htf_div = htf_div or ""
    ltf_div = ltf_div or ""

    if htf_div == "Regular Bullish Divergence" and ltf_div == "Regular Bullish Divergence":
        bucket = "Bullish RD + Bullish RD"
        remark = (f"{symbol}: HTF ({htf_pair_label.split('->')[-1].strip() if '->' in htf_pair_label else htf_pair_label}) "
                  f"Regular Bullish Divergence and LTF Regular Bullish Divergence (Strong Reversal Confluence)")
        recommendation = "STRONG BUY"
    elif htf_div == "Hidden Bullish Divergence" and ltf_div == "Regular Bullish Divergence":
        bucket = "Bullish HD + Bullish RD"
        remark = f"{symbol}: HTF Hidden Bullish Divergence and LTF Regular Bullish Divergence (Trend Continuation with Entry Signal)"
        recommendation = "BUY"
    elif htf_div == "Regular Bearish Divergence" and ltf_div == "Regular Bearish Divergence":
        bucket = "Bearish RD + Bearish RD"
        remark = f"{symbol}: HTF Regular Bearish Divergence and LTF Regular Bearish Divergence (Strong Distribution Signal)"
        recommendation = "STRONG SELL"
    elif htf_div == "Hidden Bearish Divergence" and ltf_div == "Regular Bearish Divergence":
        bucket = "Bearish HD + Bearish RD"
        remark = f"{symbol}: HTF Hidden Bearish Divergence and LTF Regular Bearish Divergence (Downtrend Continuation Signal)"
        recommendation = "SELL"
    else:
        bucket = "Other"
        remark = ""
        recommendation = "WATCH"

    return bucket, remark, recommendation


# --- Timestamp Helpers ---
def _get_ts_col(df):
    if 'datetime' in df.columns:
        return 'datetime'
    if 'date' in df.columns:
        return 'date'
    return None

def format_ist(ts):
    """Format a timestamp as an IST string. Assumes naive timestamps are
    already local (IST) NSE feed times; tz-aware timestamps are converted."""
    if ts is None:
        return "N/A"
    ts = pd.to_datetime(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(IST)
    return ts.strftime('%d %b %Y, %I:%M %p IST')


# --- MACD 360 FNO (Daily-only, last N sessions) ---
def compute_macd360_fno(days=MACD360_DAYS):
    """
    Walks every symbol's Daily data, and for each of the last `days` daily
    sessions counts (across the whole universe):
      - MACD > 0 vs MACD < 0
      - PCO (macd>signal) vs NCO (macd<=signal)
      - MACD>0 & PCO   vs   MACD>0 & NCO
      - MACD<0 & PCO   vs   MACD<0 & NCO
    Returns a list of dicts sorted chronologically, one per session.
    """
    folder = TF_FOLDERS['D']
    if not os.path.exists(folder):
        return []

    agg = defaultdict(lambda: {
        'above_zero': 0, 'below_zero': 0,
        'pco': 0, 'nco': 0,
        'above_pco': 0, 'above_nco': 0,
        'below_pco': 0, 'below_nco': 0,
    })
    real_dates = {}

    symbols = [f.replace('.json', '') for f in os.listdir(folder) if f.endswith('.json')]

    for symbol in symbols:
        file_path = os.path.join(folder, f"{symbol}.json")
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
            df = pd.DataFrame(raw_data)
            df = calculate_indicators(df)
            if df is None or len(df) < days:
                continue

            date_col = _get_ts_col(df)
            df_tail = df.tail(days)

            for _, row in df_tail.iterrows():
                macd_v, sig_v = row.get('macd'), row.get('signal')
                if pd.isna(macd_v) or pd.isna(sig_v):
                    continue

                if date_col:
                    ts = pd.to_datetime(row[date_col])
                    key = ts.strftime('%Y-%m-%d')
                    label = ts.strftime('%d %b')
                else:
                    key = str(row.name)
                    label = key

                real_dates[key] = label

                above = macd_v > 0
                pco = macd_v > sig_v
                bucket = agg[key]

                if above:
                    bucket['above_zero'] += 1
                    if pco:
                        bucket['above_pco'] += 1
                    else:
                        bucket['above_nco'] += 1
                else:
                    bucket['below_zero'] += 1
                    if pco:
                        bucket['below_pco'] += 1
                    else:
                        bucket['below_nco'] += 1

                if pco:
                    bucket['pco'] += 1
                else:
                    bucket['nco'] += 1
        except Exception:
            continue

    sorted_keys = sorted(agg.keys())[-days:]
    result = []
    for k in sorted_keys:
        row = {'date': real_dates.get(k, k)}
        row.update(agg[k])
        result.append(row)
    return result


# --- Main Data Processing Pipeline ---
def process_stock_data():
    """
    Returns (macd_results, divergence_results, last_15m_ts):
      macd_results       -> Tab 1 (Stock Screener) rows
      divergence_results -> Tab 2 (Divergence Scanner) rows
      last_15m_ts        -> pandas Timestamp of the latest 15m candle seen
    """
    macd_results = []
    divergence_results = []
    last_15m_ts = None

    sample_folder = TF_FOLDERS['D']
    if not os.path.exists(sample_folder):
        print(f"Directory {sample_folder} not found. Please check paths.")
        return [], [], None

    symbols = [f.replace('.json', '') for f in os.listdir(sample_folder) if f.endswith('.json')]

    for symbol in symbols:
        tf_data = {}

        for tf, folder in TF_FOLDERS.items():
            file_path = os.path.join(folder, f"{symbol}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        raw_data = json.load(f)
                        df = pd.DataFrame(raw_data)
                        df = calculate_indicators(df)

                        if df is not None and len(df) >= 2:
                            macd = df['macd']
                            sig = df['signal']
                            rsi = df['rsi']

                            prev_num, prev_txt = get_macd_state(macd.iloc[-2], sig.iloc[-2])
                            curr_num, curr_txt = get_macd_state(macd.iloc[-1], sig.iloc[-1])
                            divergence_txt = detect_divergence(df)

                            tf_data[tf] = {
                                'prev_num': prev_num,
                                'curr_num': curr_num,
                                'macd_state_txt': curr_txt,
                                'divergence': divergence_txt,
                                'rsi': round(float(rsi.iloc[-1]), 2) if pd.notna(rsi.iloc[-1]) else "N/A",
                                'close': float(df['close'].iloc[-1])
                            }

                            if tf == '15':
                                date_col = _get_ts_col(df)
                                if date_col:
                                    ts = pd.to_datetime(df[date_col].iloc[-1])
                                    if last_15m_ts is None or ts > last_15m_ts:
                                        last_15m_ts = ts
                except Exception:
                    continue

        # Process TF Pairs
        for ltf_key, htf_key, pair_label in TF_PAIRS:
            if ltf_key in tf_data and htf_key in tf_data:
                ltf_info = tf_data[ltf_key]
                htf_info = tf_data[htf_key]

                # --- MACD transition (Tab 1) ---
                condition = map_transition(ltf_info['prev_num'], ltf_info['curr_num'], htf_info['curr_num'])
                if condition:
                    macd_results.append({
                        'symbol': symbol,
                        'pair': pair_label,
                        'close': ltf_info['close'],
                        'htf_state': htf_info['macd_state_txt'],
                        'ltf_state': ltf_info['macd_state_txt'],
                        'htf_rsi': htf_info['rsi'],
                        'ltf_rsi': ltf_info['rsi'],
                        'condition': condition
                    })

                # --- Divergence (Tab 2) ---
                ltf_div = ltf_info['divergence']
                htf_div = htf_info['divergence']
                if ltf_div or htf_div:
                    div_type = ltf_div or htf_div

                    bucket, remark, recommendation = get_divergence_recommendation(
                        symbol, pair_label, pair_label, htf_div, ltf_div
                    )

                    divergence_results.append({
                        'symbol': symbol,
                        'pair': pair_label,
                        'close': ltf_info['close'],
                        'htf_divergence': htf_div or "",
                        'ltf_divergence': ltf_div or "",
                        'htf_rsi': htf_info['rsi'],
                        'ltf_rsi': ltf_info['rsi'],
                        'type': div_type,
                        'bucket': bucket,
                        'remark': remark,
                        'recommendation': recommendation
                    })

    # Group/sort Timeframe-Pair wise by default
    macd_results.sort(key=lambda r: (PAIR_ORDER.get(r['pair'], 99), r['symbol']))
    divergence_results.sort(key=lambda r: (PAIR_ORDER.get(r['pair'], 99), r['symbol']))

    return macd_results, divergence_results, last_15m_ts


# --- Generate Embedded HTML Dashboard ---
def build_html_dashboard(macd_results, divergence_results, macd360_data, last_15m_ts, date_str):
    json_data = json.dumps(macd_results)
    div_json_data = json.dumps(divergence_results)
    guide_json_data = json.dumps(MARKET_BIAS_GUIDE)
    div_guide_json_data = json.dumps(DIVERGENCE_GUIDE)
    reco_guide_json_data = json.dumps(RECOMMENDATION_GUIDE)
    macd360_json_data = json.dumps(macd360_data)
    last_updated_str = format_ist(last_15m_ts) if last_15m_ts is not None else "N/A"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MACD Master // Multi-Timeframe Scanner // RaoSab.in</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg-void:#060a14;
            --bg-deep:#0b1120;
            --panel-glass: rgba(17,25,45,0.72);
            --border: rgba(255,255,255,0.08);
            --border-bright: rgba(0,229,255,0.35);
            --cyan:#00e5ff;
            --cyan-soft: rgba(0,229,255,0.15);
            --magenta:#ff3d81;
            --amber:#ffb703;
            --green:#00ffa3;
            --green-soft: rgba(0,255,163,0.14);
            --red:#ff4d5e;
            --red-soft: rgba(255,77,94,0.14);
            --amber-soft: rgba(255,183,3,0.14);
            --text-main:#eef2ff;
            --text-dim:#8892b0;
            --text-faint:#5b6584;
            --font-display:'Sora', sans-serif;
            --font-body:'Inter', sans-serif;
            --font-mono:'JetBrains Mono', monospace;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin:0; min-height:100vh; font-family: var(--font-body); color: var(--text-main);
            background:
              radial-gradient(ellipse 1200px 600px at 15% -10%, rgba(0,229,255,0.10), transparent 60%),
              radial-gradient(ellipse 1000px 700px at 110% 10%, rgba(255,61,129,0.10), transparent 55%),
              radial-gradient(ellipse 900px 900px at 50% 120%, rgba(0,255,163,0.06), transparent 60%),
              var(--bg-void);
            padding: 20px 20px 60px; position: relative;
        }}
        body::before {{
            content:""; position: fixed; inset:0; pointer-events:none;
            background-image: linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
                               linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: radial-gradient(ellipse 1200px 700px at 50% 0%, black, transparent 75%);
            z-index:0;
        }}
        .container {{ max-width: 1340px; margin: 0 auto; position: relative; z-index:1; }}
        .brand-bar {{ display:flex; align-items:center; justify-content:space-between; padding: 8px 4px 16px; margin-bottom: 6px; }}
        .brand-logo {{ display:flex; align-items:center; gap:9px; text-decoration:none; }}
        .brand-mark {{
            width:26px; height:26px; border-radius:7px;
            background: linear-gradient(135deg, var(--cyan), var(--magenta));
            display:flex; align-items:center; justify-content:center;
            font-family: var(--font-display); font-weight:800; font-size:13px; color:#060a14;
        }}
        .brand-name {{ font-family: var(--font-display); font-weight:700; font-size: 15px; letter-spacing: 0.5px; color: var(--text-main); }}
        .brand-name span {{ color: var(--cyan); }}
        .brand-tag {{ font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); letter-spacing: 1px; }}
        header {{
            display:flex; align-items:flex-end; justify-content:space-between; flex-wrap:wrap; gap:16px;
            margin-bottom: 28px; padding-bottom: 22px; border-bottom: 1px solid var(--border);
        }}
        .brand-eyebrow {{
            display:flex; align-items:center; gap:8px; font-family: var(--font-mono); font-size: 11px;
            letter-spacing: 2.5px; color: var(--cyan); text-transform: uppercase; margin-bottom: 10px;
        }}
        .pulse-dot {{
            width:8px; height:8px; border-radius:50%; background: var(--green);
            box-shadow: 0 0 0 0 rgba(0,255,163,0.7); animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%{{ box-shadow: 0 0 0 0 rgba(0,255,163,0.55);}}
            70%{{ box-shadow: 0 0 0 8px rgba(0,255,163,0);}}
            100%{{ box-shadow: 0 0 0 0 rgba(0,255,163,0);}}
        }}
        h1 {{
            font-family: var(--font-display); font-weight: 800; font-size: clamp(28px, 3.4vw, 42px);
            line-height:1.05; margin:0; letter-spacing: -0.5px;
            background: linear-gradient(100deg, #ffffff 10%, var(--cyan) 55%, var(--magenta) 100%);
            -webkit-background-clip:text; background-clip:text; color:transparent;
        }}
        .subtitle {{ font-family: var(--font-body); color: var(--text-dim); margin: 10px 0 0; font-size: 14.5px; max-width: 540px; }}
        .last-updated {{
            font-family: var(--font-mono); font-size: 11.5px; color: var(--text-faint); margin-top: 10px;
            display:flex; align-items:center; gap:7px; letter-spacing: 0.3px;
        }}
        .last-updated b {{ color: var(--cyan); font-weight: 600; }}
        .header-stats {{ display:flex; gap:10px; flex-wrap:wrap; }}
        .mini-stat {{
            font-family: var(--font-mono); background: var(--panel-glass); border: 1px solid var(--border);
            border-radius: 10px; padding: 10px 16px; text-align:right; min-width: 108px; backdrop-filter: blur(6px);
            cursor: pointer; transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }}
        .mini-stat:hover {{ transform: translateY(-2px); border-color: var(--border-bright); box-shadow: 0 8px 24px -10px rgba(0,229,255,0.35); }}
        .mini-stat .k {{ display:block; font-size:10px; letter-spacing:1.5px; color:var(--text-faint); text-transform:uppercase; margin-bottom:4px;}}
        .mini-stat .v {{ display:block; font-size:17px; font-weight:700; }}
        .mini-stat.up .v {{ color: var(--green); }}
        .mini-stat.down .v {{ color: var(--red); }}
        .mini-stat.flat .v {{ color: var(--amber); }}
        .mini-stat.cyan .v {{ color: var(--cyan); }}
        .mini-stat .hint {{ display:block; font-size:9px; color: var(--text-faint); margin-top:3px; letter-spacing: 0.5px; }}
        .stat-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom: 20px; }}
        .tab-buttons {{ display:flex; gap:6px; margin-bottom: 22px; position: relative; flex-wrap: wrap; }}
        .tab-btn {{
            font-family: var(--font-display); padding: 13px 26px; font-weight: 700; font-size: 14px;
            letter-spacing: 0.3px; cursor: pointer; border: 1px solid var(--border); border-bottom: none;
            background: var(--panel-glass); color: var(--text-dim); border-radius: 10px 10px 0 0;
            transition: all 0.25s cubic-bezier(.4,0,.2,1); position: relative; backdrop-filter: blur(6px);
        }}
        .tab-btn:hover {{ color: var(--text-main); background: rgba(0,229,255,0.06); }}
        .tab-btn.active {{
            background: linear-gradient(180deg, rgba(0,229,255,0.16), rgba(0,229,255,0.03));
            color: var(--cyan); border-color: var(--border-bright); box-shadow: 0 -2px 18px rgba(0,229,255,0.18);
        }}
        .tab-btn.active::after {{ content:""; position:absolute; left:0; right:0; bottom:-1px; height:2px; background: linear-gradient(90deg, var(--cyan), var(--magenta)); }}
        .tab-content {{ display:none; animation: fadeUp 0.35s ease; }}
        .tab-content.active {{ display:block; }}
        @keyframes fadeUp {{ from{{ opacity:0; transform: translateY(6px);}} to{{ opacity:1; transform: translateY(0);}} }}
        .controls {{
            display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:18px;
            background: linear-gradient(155deg, var(--panel-glass), rgba(15,23,42,0.4)); border: 1px solid var(--border);
            padding: 22px; border-radius: 14px; margin-bottom: 22px; backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px -12px rgba(0,0,0,0.6);
        }}
        .control-group {{ display:flex; flex-direction:column; gap:9px; }}
        label {{
            font-family: var(--font-mono); font-weight:600; font-size: 11px; letter-spacing: 1.4px;
            text-transform: uppercase; color: var(--cyan); display:flex; align-items:center; gap:8px;
        }}
        label::before {{ content:""; width:5px; height:5px; border-radius:50%; background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }}
        select {{
            padding: 12px 14px; background: var(--bg-deep); color: var(--text-main); border: 1px solid var(--border);
            border-radius: 8px; font-family: var(--font-mono); font-size: 14px; font-weight:500; outline: none; cursor: pointer;
            transition: border-color 0.2s, box-shadow 0.2s; appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2300e5ff' stroke-width='3'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat; background-position: right 14px center; padding-right: 36px;
        }}
        select:focus {{ border-color: var(--cyan); box-shadow: 0 0 0 3px var(--cyan-soft); }}
        .stats-badge {{
            display:inline-flex; align-items:center; gap:10px;
            background: linear-gradient(120deg, rgba(0,229,255,0.18), rgba(255,61,129,0.12));
            border: 1px solid var(--border-bright); color: var(--text-main); padding: 12px 22px; border-radius: 10px;
            font-family: var(--font-display); font-weight: 700; font-size: 15px; margin-bottom: 20px;
            box-shadow: 0 0 24px rgba(0,229,255,0.12);
        }}
        .stats-badge::before {{ content:"◆"; color: var(--cyan); font-size: 12px; }}
        .stats-badge .count-num {{ font-family: var(--font-mono); color: var(--cyan); font-size: 17px; }}
        .table-wrap {{
            background: var(--panel-glass); border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
            backdrop-filter: blur(10px); box-shadow: 0 14px 40px -16px rgba(0,0,0,0.7);
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 14px 18px; text-align: left; }}
        th {{
            background: linear-gradient(180deg, rgba(0,229,255,0.08), rgba(0,229,255,0.02)); color: var(--cyan);
            font-family: var(--font-mono); font-weight: 600; text-transform: uppercase; font-size: 11px;
            letter-spacing: 1px; border-bottom: 1px solid var(--border-bright); white-space: nowrap;
        }}
        th.sortable {{ cursor:pointer; user-select:none; }}
        th.sortable:hover {{ color: #fff; }}
        th.sortable .arrow {{ display:inline-block; margin-left:5px; opacity:0.5; font-size:9px; }}
        th.sortable.sort-asc .arrow, th.sortable.sort-desc .arrow {{ opacity:1; color: var(--magenta); }}
        td {{ font-family: var(--font-mono); font-size: 13.5px; color: var(--text-main); }}
        tbody tr {{ border-bottom: 1px solid var(--border); transition: background 0.15s ease; }}
        tbody tr:hover {{ background: linear-gradient(90deg, rgba(0,229,255,0.06), rgba(255,61,129,0.03)); }}
        tbody tr:last-child {{ border-bottom: none; }}
        .sym {{ font-family: var(--font-display); font-weight: 700; color: #ffffff; letter-spacing: 0.3px; }}
        .pair-chip {{ display:inline-block; font-size: 11px; padding: 3px 9px; border-radius: 6px; background: rgba(255,255,255,0.06); color: var(--text-dim); border: 1px solid var(--border); }}
        .badge {{ display:inline-block; padding: 5px 12px; border-radius: 20px; font-family: var(--font-body); font-size: 11.5px; font-weight: 700; letter-spacing: 0.2px; border: 1px solid transparent; white-space: nowrap; }}
        .bullish {{ background: var(--green-soft); color: var(--green); border-color: rgba(0,255,163,0.35); box-shadow: 0 0 12px rgba(0,255,163,0.12); }}
        .bearish {{ background: var(--red-soft); color: var(--red); border-color: rgba(255,77,94,0.35); box-shadow: 0 0 12px rgba(255,77,94,0.12); }}
        .neutral {{ background: var(--amber-soft); color: var(--amber); border-color: rgba(255,183,3,0.35); box-shadow: 0 0 12px rgba(255,183,3,0.12); }}
        .reco-strongbuy {{ background: var(--green-soft); color: var(--green); border-color: rgba(0,255,163,0.5); }}
        .reco-buy {{ background: rgba(0,255,163,0.08); color: var(--green); border-color: rgba(0,255,163,0.25); }}
        .reco-sell {{ background: rgba(255,77,94,0.08); color: var(--red); border-color: rgba(255,77,94,0.25); }}
        .reco-strongsell {{ background: var(--red-soft); color: var(--red); border-color: rgba(255,77,94,0.5); }}
        .reco-watch {{ background: rgba(255,255,255,0.05); color: var(--text-dim); border-color: var(--border); }}
        .empty-row td {{ text-align:center; color: var(--text-faint); padding: 50px 20px; font-family: var(--font-body); font-size: 14px; }}
        .guide-intro {{ font-family: var(--font-display); font-weight: 800; font-size: 24px; margin: 0 0 6px; color: var(--text-main); }}
        .guide-sub {{ color: var(--text-dim); font-size: 13.5px; margin: 0 0 20px; }}
        .guide-table td:first-child {{ width: 230px; }}
        .guide-table td:last-child {{ font-family: var(--font-body); color: var(--text-dim); line-height: 1.6; font-size: 13.5px; }}
        .section-gap {{ margin-top: 36px; }}
        .chart-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
        .chart-card {{
            background: var(--panel-glass); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px 10px;
            backdrop-filter: blur(10px); box-shadow: 0 14px 40px -16px rgba(0,0,0,0.7);
        }}
        .chart-card h3 {{ font-family: var(--font-display); font-size: 15px; margin: 0 0 4px; color: var(--text-main); }}
        .chart-card p {{ font-family: var(--font-body); font-size: 12px; color: var(--text-dim); margin: 0 0 12px; }}
        .chart-card canvas {{ max-height: 260px; }}
        footer {{ text-align:center; color: var(--text-faint); font-family: var(--font-mono); font-size: 11px; letter-spacing: 1px; margin-top: 34px; text-transform: uppercase; }}
        footer a {{ color: var(--cyan); text-decoration:none; }}
        footer a:hover {{ text-decoration:underline; }}
        .modal-overlay {{ display:none; position: fixed; inset:0; background: rgba(3,6,14,0.78); backdrop-filter: blur(4px); z-index: 50; align-items: center; justify-content: center; padding: 24px; }}
        .modal-overlay.active {{ display:flex; }}
        .modal-panel {{
            width: 100%; max-width: 1100px; max-height: 82vh; background: linear-gradient(155deg, #0d1526, #0a0f1d);
            border: 1px solid var(--border-bright); border-radius: 16px;
            box-shadow: 0 30px 80px -20px rgba(0,0,0,0.8), 0 0 40px rgba(0,229,255,0.08);
            display:flex; flex-direction:column; overflow:hidden; animation: modalIn 0.25s ease;
        }}
        @keyframes modalIn {{ from{{ opacity:0; transform: translateY(14px) scale(0.98);}} to{{ opacity:1; transform: translateY(0) scale(1);}} }}
        .modal-header {{ display:flex; align-items:center; justify-content:space-between; padding: 18px 24px; border-bottom: 1px solid var(--border); }}
        .modal-title {{ font-family: var(--font-display); font-weight: 800; font-size: 19px; display:flex; align-items:center; gap:10px; }}
        .modal-title .badge {{ font-size: 12px; }}
        .modal-close {{
            background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: var(--text-dim);
            width: 34px; height: 34px; border-radius: 9px; cursor:pointer; font-size: 16px; font-family: var(--font-mono);
            transition: all 0.15s ease;
        }}
        .modal-close:hover {{ color: var(--red); border-color: rgba(255,77,94,0.4); background: rgba(255,77,94,0.08); }}
        .modal-body {{ overflow-y: auto; padding: 0; }}
        .modal-body table th {{ position: sticky; top:0; z-index:2; }}
        .modal-body::-webkit-scrollbar {{ width:8px; }}
        .modal-body::-webkit-scrollbar-thumb {{ background: rgba(0,229,255,0.25); border-radius: 8px; }}
        .modal-body::-webkit-scrollbar-track {{ background: transparent; }}
        @media (max-width: 640px) {{
            .controls {{ grid-template-columns: 1fr; }}
            th, td {{ padding: 10px 12px; font-size: 12px; }}
            .tab-btn {{ padding: 11px 16px; font-size: 12.5px; }}
        }}
    </style>
</head>
<body>

<div class="container">

    <div class="brand-bar">
        <a class="brand-logo" href="https://www.raosab.in" target="_blank" rel="noopener">
            <span class="brand-mark">RS</span>
            <span class="brand-name">RAO<span>SAB</span>.IN</span>
        </a>
        <span class="brand-tag">Technical Scanner Suite</span>
    </div>

    <header>
        <div>
            <div class="brand-eyebrow"><span class="pulse-dot"></span> LIVE SCAN &middot; MULTI-TIMEFRAME ENGINE</div>
            <h1>MACD Master Dashboard</h1>
            <p class="subtitle">Automated transition detection, MACD state mapping, RSI(14) correlation &amp; divergence scanning across four timeframe pairs.</p>
            <div class="last-updated">&#9201; Last Data Updated (15m candle, IST): <b id="lastUpdatedTime">{last_updated_str}</b></div>
        </div>
        <div class="header-stats">
            <div class="mini-stat up" onclick="openBiasModal('bullish')">
                <span class="k">Bullish Setups</span><span class="v" id="statBull">0</span>
                <span class="hint">tap to view all &rarr;</span>
            </div>
            <div class="mini-stat down" onclick="openBiasModal('bearish')">
                <span class="k">Bearish Setups</span><span class="v" id="statBear">0</span>
                <span class="hint">tap to view all &rarr;</span>
            </div>
            <div class="mini-stat flat" onclick="openBiasModal('neutral')">
                <span class="k">Neutral</span><span class="v" id="statNeutral">0</span>
                <span class="hint">tap to view all &rarr;</span>
            </div>
        </div>
    </header>

    <div class="tab-buttons">
        <button class="tab-btn active" onclick="switchTab('tabScreener')">&#9889; Stock Screener</button>
        <button class="tab-btn" onclick="switchTab('tabDivergence')">&#127760; Divergence Scanner</button>
        <button class="tab-btn" onclick="switchTab('tabGuide')">&#128214; Market Bias Guide</button>
        <button class="tab-btn" onclick="switchTab('tabMacd360')">&#128207; MACD 360 FNO</button>
    </div>

    <!-- TAB 1: SCREENER -->
    <div id="tabScreener" class="tab-content active">
        <div class="controls">
            <div class="control-group">
                <label for="pairSelect">Timeframe Pair (LTF &rarr; HTF)</label>
                <select id="pairSelect" onchange="filterData()">
                    <option value="15m -> 1h">15m &rarr; 1h</option>
                    <option value="1h -> Daily">1h &rarr; Daily</option>
                    <option value="Daily -> Weekly">Daily &rarr; Weekly</option>
                    <option value="Weekly -> Monthly">Weekly &rarr; Monthly</option>
                </select>
            </div>
            <div class="control-group">
                <label for="conditionSelect">MACD Transition Condition</label>
                <select id="conditionSelect" onchange="filterData()">
                    <option value="Strong Bullish">Strong Bullish</option>
                    <option value="Bullish Reversal">Bullish Reversal</option>
                    <option value="Bullish Pullback">Bullish Pullback</option>
                    <option value="Deep Pullback">Deep Pullback</option>
                    <option value="Dip Buy / Reversal">Dip Buy / Reversal</option>
                    <option value="Oversold Accumulation">Oversold Accumulation</option>
                    <option value="Neutral / Consolidating">Neutral / Consolidating</option>
                    <option value="Complex Reversal Failure">Complex Reversal Failure</option>
                    <option value="Counter-Trend Buy">Counter-Trend Buy</option>
                    <option value="Early Recovery Attempt">Early Recovery Attempt</option>
                    <option value="Correction in Progress">Correction in Progress</option>
                    <option value="Accelerating Correction">Accelerating Correction</option>
                    <option value="Aggressive Counter-Trend">Aggressive Counter-Trend</option>
                    <option value="Weak Oversold Bounce">Weak Oversold Bounce</option>
                    <option value="Bearish Continuation">Bearish Continuation</option>
                    <option value="Strong Bearish">Strong Bearish</option>
                </select>
            </div>
        </div>

        <div class="stats-badge">Matching Stocks: <span class="count-num" id="countBadge">0</span></div>

        <div class="table-wrap">
            <table id="screenerTable">
                <thead>
                    <tr>
                        <th class="sortable" data-key="symbol">Symbol<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="pair">Timeframe Pair<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="close">Close Price<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="htf_state">HTF MACD State<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="ltf_state">LTF MACD State<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="htf_rsi">HTF RSI (14)<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="ltf_rsi">LTF RSI (14)<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="condition">Market Bias<span class="arrow">&#9650;&#9660;</span></th>
                    </tr>
                </thead>
                <tbody id="stockTableBody"></tbody>
            </table>
        </div>
    </div>

    <!-- TAB 2: DIVERGENCE SCANNER -->
    <div id="tabDivergence" class="tab-content">
        <div class="stat-row">
            <div class="mini-stat up"><span class="k">Strong Buy</span><span class="v" id="statStrongBuy">0</span></div>
            <div class="mini-stat up"><span class="k">Buy</span><span class="v" id="statBuy">0</span></div>
            <div class="mini-stat down"><span class="k">Sell</span><span class="v" id="statSell">0</span></div>
            <div class="mini-stat down"><span class="k">Strong Sell</span><span class="v" id="statStrongSell">0</span></div>
            <div class="mini-stat flat"><span class="k">Watch</span><span class="v" id="statWatch">0</span></div>
        </div>
        <div class="controls">
            <div class="control-group">
                <label for="divPairSelect">Timeframe Pair (LTF &rarr; HTF)</label>
                <select id="divPairSelect" onchange="filterDivergenceData()">
                    <option value="15m -> 1h">15m &rarr; 1h</option>
                    <option value="1h -> Daily">1h &rarr; Daily</option>
                    <option value="Daily -> Weekly">Daily &rarr; Weekly</option>
                    <option value="Weekly -> Monthly">Weekly &rarr; Monthly</option>
                </select>
            </div>
            <div class="control-group">
                <label for="divTypeSelect">Divergence Type</label>
                <select id="divTypeSelect" onchange="filterDivergenceData()">
                    <option value="Regular Bullish Divergence">Regular Bullish Divergence</option>
                    <option value="Regular Bearish Divergence">Regular Bearish Divergence</option>
                    <option value="Hidden Bullish Divergence">Hidden Bullish Divergence</option>
                    <option value="Hidden Bearish Divergence">Hidden Bearish Divergence</option>
                </select>
            </div>
            <div class="control-group">
                <label for="divRecoSelect">Recommendation</label>
                <select id="divRecoSelect" onchange="filterDivergenceData()">
                    <option value="ALL">All</option>
                    <option value="STRONG BUY">Strong Buy</option>
                    <option value="BUY">Buy</option>
                    <option value="SELL">Sell</option>
                    <option value="STRONG SELL">Strong Sell</option>
                    <option value="WATCH">Watch</option>
                </select>
            </div>
        </div>

        <div class="stats-badge">Matching Stocks: <span class="count-num" id="divCountBadge">0</span></div>

        <div class="table-wrap">
            <table id="divergenceTable">
                <thead>
                    <tr>
                        <th class="sortable" data-key="symbol">Symbol<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="pair">Timeframe Pair<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="close">Close Price<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="htf_divergence">HTF Divergence<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="ltf_divergence">LTF Divergence<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="htf_rsi">HTF RSI (14)<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="ltf_rsi">LTF RSI (14)<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="type">Type<span class="arrow">&#9650;&#9660;</span></th>
                        <th class="sortable" data-key="recommendation">Recommendation<span class="arrow">&#9650;&#9660;</span></th>
                    </tr>
                </thead>
                <tbody id="divTableBody"></tbody>
            </table>
        </div>
    </div>

    <!-- TAB 3: MARKET BIAS GUIDE -->
    <div id="tabGuide" class="tab-content">
        <p class="guide-intro">Market Bias Reference</p>
        <p class="guide-sub">Practical trading interpretation for every MACD transition condition tracked by the scanner.</p>
        <div class="table-wrap">
            <table class="guide-table">
                <thead><tr><th style="width:25%;">Market Bias</th><th style="width:75%;">Practical Trading Interpretation</th></tr></thead>
                <tbody id="guideTableBody"></tbody>
            </table>
        </div>

        <div class="section-gap">
            <p class="guide-intro" style="font-size:20px;">Divergence Types Reference</p>
            <p class="guide-sub">How the 4 divergence types are read across the LTF / HTF pair.</p>
            <div class="table-wrap">
                <table class="guide-table">
                    <thead><tr><th style="width:25%;">Divergence Type</th><th style="width:75%;">Practical Trading Interpretation</th></tr></thead>
                    <tbody id="divGuideTableBody"></tbody>
                </table>
            </div>
        </div>

        <div class="section-gap">
            <p class="guide-intro" style="font-size:20px;">Divergence Recommendation Logic</p>
            <p class="guide-sub">How HTF + LTF divergence combinations map to a call.</p>
            <div class="table-wrap">
                <table class="guide-table">
                    <thead><tr><th style="width:16%;">Call</th><th style="width:28%;">HTF + LTF Combination</th><th style="width:56%;">Why</th></tr></thead>
                    <tbody id="recoGuideTableBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- TAB 4: MACD 360 FNO (Daily-only) -->
    <div id="tabMacd360" class="tab-content">
        <p class="guide-intro">MACD 360 FNO &mdash; Daily Timeframe Breadth</p>
        <p class="guide-sub">Universe-wide Daily MACD breadth over the last {MACD360_DAYS} sessions. Counts are computed purely on the Daily timeframe (no LTF/HTF pairing).</p>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>MACD &gt; 0 vs MACD &lt; 0 (Count)</h3>
                <p>How many stocks in the universe have Daily MACD above vs below the zero line, each session.</p>
                <canvas id="chartZero"></canvas>
            </div>
            <div class="chart-card">
                <h3>MACD PCO vs NCO (Count)</h3>
                <p>Positive Crossover (MACD &gt; Signal) vs Negative Crossover (MACD &le; Signal), each session.</p>
                <canvas id="chartCross"></canvas>
            </div>
            <div class="chart-card">
                <h3>MACD &gt; 0 &amp; PCO vs MACD &gt; 0 &amp; NCO</h3>
                <p>Within the above-zero group: how many are also in a positive crossover vs a negative crossover.</p>
                <canvas id="chartAboveSplit"></canvas>
            </div>
            <div class="chart-card">
                <h3>MACD &lt; 0 &amp; PCO vs MACD &lt; 0 &amp; NCO</h3>
                <p>Within the below-zero group: how many are also in a positive crossover vs a negative crossover.</p>
                <canvas id="chartBelowSplit"></canvas>
            </div>
        </div>
    </div>

    <footer>MACD Master &middot; Multi-Timeframe Scanner &middot; {date_str} &middot; Powered by <a href="https://www.raosab.in" target="_blank" rel="noopener">RaoSab.in</a></footer>
</div>

<!-- MODAL -->
<div class="modal-overlay" id="biasModal" onclick="if(event.target===this) closeBiasModal()">
    <div class="modal-panel">
        <div class="modal-header">
            <div class="modal-title" id="modalTitle"></div>
            <button class="modal-close" onclick="closeBiasModal()">&#10005;</button>
        </div>
        <div class="modal-body">
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th><th>Setup / Condition</th><th>Timeframe Pair</th><th>HTF MACD State</th>
                        <th>LTF MACD State</th><th>HTF RSI (14)</th><th>LTF RSI (14)</th>
                    </tr>
                </thead>
                <tbody id="modalTableBody"></tbody>
            </table>
        </div>
    </div>
</div>

<script>
    const stockData = {json_data};
    const guideData = {guide_json_data};
    const divergenceData = {div_json_data};
    const divGuideData = {div_guide_json_data};
    const recoGuideData = {reco_guide_json_data};
    const macd360Data = {macd360_json_data};

    const PAIR_ORDER = {{"15m -> 1h": 0, "1h -> Daily": 1, "Daily -> Weekly": 2, "Weekly -> Monthly": 3}};

    function switchTab(tabId) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        event.currentTarget.classList.add('active');
        if (tabId === 'tabMacd360') {{
            renderMacd360Charts();
        }}
    }}

    function getBadgeClass(cond) {{
        if (cond.includes('Bullish') || cond.includes('Buy') || cond.includes('Recovery') || cond.includes('Accumulation')) {{
            return 'bullish';
        }} else if (cond.includes('Bearish') || cond.includes('Correction') || cond.includes('Failure')) {{
            return 'bearish';
        }}
        return 'neutral';
    }}

    function getDivBadgeClass(type) {{
        return type.includes('Bullish') ? 'bullish' : type.includes('Bearish') ? 'bearish' : 'neutral';
    }}

    function getRecoClass(reco) {{
        switch (reco) {{
            case 'STRONG BUY': return 'reco-strongbuy';
            case 'BUY': return 'reco-buy';
            case 'SELL': return 'reco-sell';
            case 'STRONG SELL': return 'reco-strongsell';
            default: return 'reco-watch';
        }}
    }}

    function rsiColor(v) {{
        return v > 60 ? '#00ffa3' : v < 40 ? '#ff4d5e' : '#ffb703';
    }}

    function updateHeaderStats() {{
        let bull = 0, bear = 0, neu = 0;
        stockData.forEach(item => {{
            const cls = getBadgeClass(item.condition);
            if (cls === 'bullish') bull++;
            else if (cls === 'bearish') bear++;
            else neu++;
        }});
        document.getElementById('statBull').innerText = bull;
        document.getElementById('statBear').innerText = bear;
        document.getElementById('statNeutral').innerText = neu;
    }}

    function updateDivergenceStats() {{
        let sb = 0, b = 0, s = 0, ss = 0, w = 0;
        divergenceData.forEach(item => {{
            switch (item.recommendation) {{
                case 'STRONG BUY': sb++; break;
                case 'BUY': b++; break;
                case 'SELL': s++; break;
                case 'STRONG SELL': ss++; break;
                default: w++;
            }}
        }});
        document.getElementById('statStrongBuy').innerText = sb;
        document.getElementById('statBuy').innerText = b;
        document.getElementById('statSell').innerText = s;
        document.getElementById('statStrongSell').innerText = ss;
        document.getElementById('statWatch').innerText = w;
    }}

    function openBiasModal(type) {{
        const labels = {{ bullish: 'Bullish Setups', bearish: 'Bearish Setups', neutral: 'Neutral Setups' }};
        const matches = stockData.filter(item => getBadgeClass(item.condition) === type);

        document.getElementById('modalTitle').innerHTML =
            `${{labels[type]}} <span class="badge ${{type}}">${{matches.length}} Stocks</span>`;

        const body = document.getElementById('modalTableBody');
        body.innerHTML = '';

        if (matches.length === 0) {{
            body.innerHTML = `<tr class="empty-row"><td colspan="7">No stocks currently in this category.</td></tr>`;
        }} else {{
            matches
                .slice()
                .sort((a, b) => (PAIR_ORDER[a.pair] - PAIR_ORDER[b.pair]) || a.symbol.localeCompare(b.symbol))
                .forEach(row => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="sym">${{row.symbol}}</td>
                    <td><span class="badge ${{type}}">${{row.condition}}</span></td>
                    <td><span class="pair-chip">${{row.pair}}</span></td>
                    <td style="color:#cbd5e1;">${{row.htf_state}}</td>
                    <td style="color:#cbd5e1;">${{row.ltf_state}}</td>
                    <td style="font-weight:700; color:${{rsiColor(row.htf_rsi)}};">${{row.htf_rsi}}</td>
                    <td style="font-weight:700; color:${{rsiColor(row.ltf_rsi)}};">${{row.ltf_rsi}}</td>
                `;
                body.appendChild(tr);
            }});
        }}
        document.getElementById('biasModal').classList.add('active');
    }}

    function closeBiasModal() {{
        document.getElementById('biasModal').classList.remove('active');
    }}

    document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') closeBiasModal(); }});

    let screenerSort = {{ key: 'pair', asc: true }};
    let divergenceSort = {{ key: 'pair', asc: true }};

    function sortRows(rows, sortState) {{
        const {{ key, asc }} = sortState;
        const sorted = rows.slice().sort((a, b) => {{
            let av = a[key], bv = b[key];
            if (key === 'pair') {{ av = PAIR_ORDER[av]; bv = PAIR_ORDER[bv]; }}
            if (typeof av === 'number' && typeof bv === 'number') {{
                return asc ? av - bv : bv - av;
            }}
            av = String(av); bv = String(bv);
            return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        }});
        return sorted;
    }}

    function bindSortableHeaders(tableId, sortStateGetter, sortStateSetter, rerenderFn) {{
        document.querySelectorAll(`#${{tableId}} th.sortable`).forEach(th => {{
            th.addEventListener('click', () => {{
                const key = th.getAttribute('data-key');
                const state = sortStateGetter();
                const asc = (state.key === key) ? !state.asc : true;
                sortStateSetter({{ key, asc }});
                document.querySelectorAll(`#${{tableId}} th.sortable`).forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                th.classList.add(asc ? 'sort-asc' : 'sort-desc');
                rerenderFn();
            }});
        }});
    }}

    function filterData() {{
        const selectedPair = document.getElementById('pairSelect').value;
        const selectedCondition = document.getElementById('conditionSelect').value;
        const tableBody = document.getElementById('stockTableBody');
        let filtered = stockData.filter(item => item.pair === selectedPair && item.condition === selectedCondition);
        filtered = sortRows(filtered, screenerSort);
        document.getElementById('countBadge').innerText = filtered.length;
        tableBody.innerHTML = '';
        if (filtered.length === 0) {{
            tableBody.innerHTML = `<tr class="empty-row"><td colspan="8">No stocks matching this pair &amp; condition.</td></tr>`;
            return;
        }}
        filtered.forEach(row => {{
            const badgeClass = getBadgeClass(row.condition);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="sym">${{row.symbol}}</td>
                <td><span class="pair-chip">${{row.pair}}</span></td>
                <td>${{row.close !== undefined ? row.close.toFixed(2) : 'N/A'}}</td>
                <td style="color:#cbd5e1;">${{row.htf_state}}</td>
                <td style="color:#cbd5e1;">${{row.ltf_state}}</td>
                <td style="font-weight:700; color:${{rsiColor(row.htf_rsi)}};">${{row.htf_rsi}}</td>
                <td style="font-weight:700; color:${{rsiColor(row.ltf_rsi)}};">${{row.ltf_rsi}}</td>
                <td><span class="badge ${{badgeClass}}">${{row.condition}}</span></td>
            `;
            tableBody.appendChild(tr);
        }});
    }}

    function filterDivergenceData() {{
        const selectedPair = document.getElementById('divPairSelect').value;
        const selectedType = document.getElementById('divTypeSelect').value;
        const selectedReco = document.getElementById('divRecoSelect').value;
        const tableBody = document.getElementById('divTableBody');
        let filtered = divergenceData.filter(item =>
            item.pair === selectedPair &&
            item.type === selectedType &&
            (selectedReco === 'ALL' || item.recommendation === selectedReco)
        );
        filtered = sortRows(filtered, divergenceSort);
        document.getElementById('divCountBadge').innerText = filtered.length;
        tableBody.innerHTML = '';
        if (filtered.length === 0) {{
            tableBody.innerHTML = `<tr class="empty-row"><td colspan="9">No divergence matching this pair, type &amp; recommendation.</td></tr>`;
            return;
        }}
        filtered.forEach(row => {{
            const badgeClass = getDivBadgeClass(row.type);
            const recoClass = getRecoClass(row.recommendation);
            const tr = document.createElement('tr');
            tr.title = row.remark || '';
            tr.innerHTML = `
                <td class="sym">${{row.symbol}}</td>
                <td><span class="pair-chip">${{row.pair}}</span></td>
                <td>${{row.close !== undefined ? row.close.toFixed(2) : 'N/A'}}</td>
                <td style="color:#cbd5e1;">${{row.htf_divergence || '&mdash;'}}</td>
                <td style="color:#cbd5e1;">${{row.ltf_divergence || '&mdash;'}}</td>
                <td style="font-weight:700; color:${{rsiColor(row.htf_rsi)}};">${{row.htf_rsi}}</td>
                <td style="font-weight:700; color:${{rsiColor(row.ltf_rsi)}};">${{row.ltf_rsi}}</td>
                <td><span class="badge ${{badgeClass}}">${{row.type}}</span></td>
                <td><span class="badge ${{recoClass}}">${{row.recommendation}}</span></td>
            `;
            tableBody.appendChild(tr);
        }});
    }}

    function populateGuide() {{
        const guideBody = document.getElementById('guideTableBody');
        guideBody.innerHTML = '';
        guideData.forEach(row => {{
            const badgeClass = getBadgeClass(row['Market Bias']);
            const tr = document.createElement('tr');
            tr.innerHTML = `<td><span class="badge ${{badgeClass}}">${{row['Market Bias']}}</span></td><td>${{row['Interpretation']}}</td>`;
            guideBody.appendChild(tr);
        }});

        const divGuideBody = document.getElementById('divGuideTableBody');
        divGuideBody.innerHTML = '';
        divGuideData.forEach(row => {{
            const badgeClass = getDivBadgeClass(row['Divergence Type']);
            const tr = document.createElement('tr');
            tr.innerHTML = `<td><span class="badge ${{badgeClass}}">${{row['Divergence Type']}}</span></td><td>${{row['Interpretation']}}</td>`;
            divGuideBody.appendChild(tr);
        }});

        const recoGuideBody = document.getElementById('recoGuideTableBody');
        recoGuideBody.innerHTML = '';
        recoGuideData.forEach(row => {{
            const recoClass = getRecoClass(row['Recommendation']);
            const tr = document.createElement('tr');
            tr.innerHTML = `<td><span class="badge ${{recoClass}}">${{row['Recommendation']}}</span></td><td style="color:#cbd5e1;">${{row['Bucket']}}</td><td>${{row['Interpretation']}}</td>`;
            recoGuideBody.appendChild(tr);
        }});
    }}

    let macd360Charts = {{}};
    function renderMacd360Charts() {{
        if (!macd360Data || macd360Data.length === 0 || typeof Chart === 'undefined') return;
        if (macd360Charts.rendered) return;
        macd360Charts.rendered = true;

        const labels = macd360Data.map(d => d.date);
        const commonOpts = {{
            responsive: true,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{ legend: {{ labels: {{ color: '#8892b0', font: {{ family: 'JetBrains Mono' }} }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#5b6584' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                y: {{ ticks: {{ color: '#5b6584' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, beginAtZero: true }}
            }}
        }};

        new Chart(document.getElementById('chartZero'), {{
            type: 'line',
            data: {{
                labels,
                datasets: [
                    {{ label: 'MACD > 0', data: macd360Data.map(d => d.above_zero), borderColor: '#00ffa3', backgroundColor: 'rgba(0,255,163,0.12)', tension: 0.3, fill: true }},
                    {{ label: 'MACD < 0', data: macd360Data.map(d => d.below_zero), borderColor: '#ff4d5e', backgroundColor: 'rgba(255,77,94,0.12)', tension: 0.3, fill: true }}
                ]
            }},
            options: commonOpts
        }});

        new Chart(document.getElementById('chartCross'), {{
            type: 'line',
            data: {{
                labels,
                datasets: [
                    {{ label: 'PCO (MACD > Signal)', data: macd360Data.map(d => d.pco), borderColor: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.12)', tension: 0.3, fill: true }},
                    {{ label: 'NCO (MACD <= Signal)', data: macd360Data.map(d => d.nco), borderColor: '#ff3d81', backgroundColor: 'rgba(255,61,129,0.12)', tension: 0.3, fill: true }}
                ]
            }},
            options: commonOpts
        }});

        new Chart(document.getElementById('chartAboveSplit'), {{
            type: 'line',
            data: {{
                labels,
                datasets: [
                    {{ label: 'MACD > 0 & PCO', data: macd360Data.map(d => d.above_pco), borderColor: '#00ffa3', backgroundColor: 'rgba(0,255,163,0.12)', tension: 0.3, fill: true }},
                    {{ label: 'MACD > 0 & NCO', data: macd360Data.map(d => d.above_nco), borderColor: '#ffb703', backgroundColor: 'rgba(255,183,3,0.12)', tension: 0.3, fill: true }}
                ]
            }},
            options: commonOpts
        }});

        new Chart(document.getElementById('chartBelowSplit'), {{
            type: 'line',
            data: {{
                labels,
                datasets: [
                    {{ label: 'MACD < 0 & PCO', data: macd360Data.map(d => d.below_pco), borderColor: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.12)', tension: 0.3, fill: true }},
                    {{ label: 'MACD < 0 & NCO', data: macd360Data.map(d => d.below_nco), borderColor: '#ff4d5e', backgroundColor: 'rgba(255,77,94,0.12)', tension: 0.3, fill: true }}
                ]
            }},
            options: commonOpts
        }});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
        bindSortableHeaders('screenerTable', () => screenerSort, (s) => {{ screenerSort = s; }}, filterData);
        bindSortableHeaders('divergenceTable', () => divergenceSort, (s) => {{ divergenceSort = s; }}, filterDivergenceData);
        filterData();
        filterDivergenceData();
        populateGuide();
        updateHeaderStats();
        updateDivergenceStats();
    }});
</script>

</body>
</html>
"""
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Successfully generated index.html dashboard with {len(macd_results)} screener rows, "
          f"{len(divergence_results)} divergence rows, and {len(macd360_data)} MACD-360 daily sessions "
          f"(last update: {last_updated_str}).")

if __name__ == '__main__':
    date_str = datetime.datetime.now(IST).strftime("%d %b %Y")
    macd_data, div_data, last_15m = process_stock_data()
    macd360_data = compute_macd360_fno(MACD360_DAYS)
    build_html_dashboard(macd_data, div_data, macd360_data, last_15m, date_str)
