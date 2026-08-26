import os
import json
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

# Market Bias Reference Dictionary for Tab 2
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

# --- Technical Indicator Functions ---
def calculate_indicators(df, fast=12, slow=26, signal=9, rsi_period=14):
    if df.empty or len(df) < slow + signal:
        return None
    
    if 'datetime' in df.columns:
        df = df.sort_values('datetime')
    elif 'date' in df.columns:
        df = df.sort_values('date')

    close = df['close']
    
    # MACD Calculation
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # RSI (14) Calculation
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
    
    if htf_curr == 1:
        if ltf_prev == 3 and ltf_curr == 1: return "Strong Bullish"
        if ltf_prev == 4 and ltf_curr == 2: return "Bullish Reversal"
        if ltf_prev == 1 and ltf_curr == 3: return "Bullish Pullback"
        if ltf_prev == 2 and ltf_curr == 4: return "Deep Pullback"
        
    elif htf_curr == 2:
        if ltf_prev == 3 and ltf_curr == 1: return "Dip Buy / Reversal"
        if ltf_prev == 4 and ltf_curr == 2: return "Oversold Accumulation"
        if ltf_prev == 1 and ltf_curr == 3: return "Neutral / Consolidating"
        if ltf_prev == 2 and ltf_curr == 4: return "Complex Reversal Failure"
        
    elif htf_curr == 3:
        if ltf_prev == 3 and ltf_curr == 1: return "Counter-Trend Buy"
        if ltf_prev == 4 and ltf_curr == 2: return "Early Recovery Attempt"
        if ltf_prev == 1 and ltf_curr == 3: return "Correction in Progress"
        if ltf_prev == 2 and ltf_curr == 4: return "Accelerating Correction"
        
    elif htf_curr == 4:
        if ltf_prev == 3 and ltf_curr == 1: return "Aggressive Counter-Trend"
        if ltf_prev == 4 and ltf_curr == 2: return "Weak Oversold Bounce"
        if ltf_prev == 1 and ltf_curr == 3: return "Bearish Continuation"
        if ltf_prev == 2 and ltf_curr == 4: return "Strong Bearish"

    return None

# --- Divergence Detection Logic ---
def detect_divergence(df, lookback=30):
    if df is None or len(df) < lookback:
        return "No Divergence"
    
    macd = df['macd']
    lows = df['low'] if 'low' in df.columns else df['close']
    highs = df['high'] if 'high' in df.columns else df['close']
    
    p_low1 = lows.iloc[-lookback:-15].min()
    p_low2 = lows.iloc[-15:].min()
    m_low1 = macd.iloc[-lookback:-15].min()
    m_low2 = macd.iloc[-15:].min()

    p_high1 = highs.iloc[-lookback:-15].max()
    p_high2 = highs.iloc[-15:].max()
    m_high1 = macd.iloc[-lookback:-15].max()
    m_high2 = macd.iloc[-15:].max()

    # Normal Divergence (ND)
    if p_low2 < p_low1 and m_low2 > m_low1:
        return "Bullish ND"
    if p_high2 > p_high1 and m_high2 < m_high1:
        return "Bearish ND"

    # Reverse Divergence (RD)
    if p_low2 > p_low1 and m_low2 < m_low1:
        return "Bullish RD"
    if p_high2 < p_high1 and m_high2 > m_high1:
        return "Bearish RD"

    return "No Divergence"

def get_setup_category(divergence_type):
    if "Bullish" in divergence_type:
        return "Bullish"
    elif "Bearish" in divergence_type:
        return "Bearish"
    return "Neutral"

# --- Main Processing Pipeline ---
def process_stock_data():
    tab1_results = []
    tab3_results = []
    
    sample_folder = TF_FOLDERS['D']
    if not os.path.exists(sample_folder):
        print(f"Directory {sample_folder} not found. Returning empty dataset.")
        return [], []

    symbols = [f.replace('.json', '') for f in os.listdir(sample_folder) if f.endswith('.json')]
    
    for symbol in symbols:
        tf_data = {}
        tf_dfs = {}
        
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
                            
                            tf_data[tf] = {
                                'prev_num': prev_num,
                                'curr_num': curr_num,
                                'macd_state_txt': curr_txt,
                                'rsi': round(float(rsi.iloc[-1]), 2) if pd.notna(rsi.iloc[-1]) else "N/A",
                                'close': float(df['close'].iloc[-1])
                            }
                            tf_dfs[tf] = df
                except Exception as e:
                    continue
        
        # Process TF Pairs
        for ltf_key, htf_key, pair_label in TF_PAIRS:
            if ltf_key in tf_data and htf_key in tf_data:
                ltf_info = tf_data[ltf_key]
                htf_info = tf_data[htf_key]
                
                # Tab 1 Logic
                condition = map_transition(ltf_info['prev_num'], ltf_info['curr_num'], htf_info['curr_num'])
                if condition:
                    tab1_results.append({
                        'symbol': symbol,
                        'pair': pair_label,
                        'close': ltf_info['close'],
                        'htf_state': htf_info['macd_state_txt'],
                        'ltf_state': ltf_info['macd_state_txt'],
                        'htf_rsi': htf_info['rsi'],
                        'ltf_rsi': ltf_info['rsi'],
                        'condition': condition
                    })

                # Tab 3 Divergence Logic
                ltf_df = tf_dfs.get(ltf_key)
                divergence = detect_divergence(ltf_df)
                category = get_setup_category(divergence)

                tab3_results.append({
                    'symbol': symbol,
                    'pair': pair_label,
                    'close': ltf_info['close'],
                    'htf_state': htf_info['macd_state_txt'],
                    'ltf_state': ltf_info['macd_state_txt'],
                    'htf_rsi': htf_info['rsi'],
                    'ltf_rsi': ltf_info['rsi'],
                    'divergence': divergence,
                    'category': category
                })
                    
    return tab1_results, tab3_results

def build_html_dashboard(tab1_data, tab3_data):
    json_tab1 = json.dumps(tab1_data)
    json_tab3 = json.dumps(tab3_data)
    guide_json_data = json.dumps(MARKET_BIAS_GUIDE)
    
    with open('index.html', 'w') as f:
        f.write(get_html_string(json_tab1, guide_json_data, json_tab3))
    print("Successfully generated index.html dashboard with Tab 1, Tab 2, and Tab 3!")

def get_html_string(json_tab1, guide_json_data, json_tab3):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MACD Master Multi-Timeframe Dashboard</title>
    <style>
        :root {{
            --bg-dark: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --text-main: #f8fafc;
            --border-color: #334155;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            margin: 0;
            padding: 20px;
        }}

        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 25px;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 15px;
        }}

        h1 {{
            color: var(--accent);
            margin: 0 0 10px 0;
            font-size: 26px;
        }}

        .tab-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}

        .tab-btn {{
            padding: 12px 24px;
            font-weight: bold;
            font-size: 15px;
            cursor: pointer;
            border: 1px solid var(--border-color);
            background: var(--card-bg);
            color: #94a3b8;
            border-radius: 8px 8px 0 0;
            transition: all 0.2s ease-in-out;
        }}

        .tab-btn.active {{
            background: var(--accent);
            color: #0f172a;
            border-color: var(--accent);
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .controls {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        label {{
            font-weight: 600;
            font-size: 14px;
            color: #94a3b8;
        }}

        select {{
            padding: 10px 14px;
            background: #0f172a;
            color: #fff;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 15px;
            outline: none;
            cursor: pointer;
        }}

        select:focus {{
            border-color: var(--accent);
        }}

        .stats-badge {{
            display: inline-block;
            background: #0284c7;
            color: white;
            padding: 10px 18px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 15px;
            margin-bottom: 20px;
        }}

        .summary-cards {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }}

        .card {{
            flex: 1;
            padding: 15px 20px;
            border-radius: 8px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            cursor: pointer;
            transition: transform 0.2s, border-color 0.2s;
        }}

        .card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent);
        }}

        .card.active {{
            border-color: var(--accent);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
        }}

        .card-title {{
            font-size: 14px;
            font-weight: 600;
            color: #94a3b8;
        }}

        .card-count {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}

        .card-bullish .card-count {{ color: #4ade80; }}
        .card-bearish .card-count {{ color: #fca5a5; }}
        .card-neutral .card-count {{ color: #fde047; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}

        th, td {{
            padding: 12px 16px;
            text-align: left;
        }}

        th {{
            background-color: #334155;
            color: var(--accent);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }}

        tr {{
            border-bottom: 1px solid var(--border-color);
        }}

        tr:hover {{
            background-color: #26334d;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}

        .bullish {{ background: #166534; color: #4ade80; }}
        .bearish {{ background: #991b1b; color: #fca5a5; }}
        .neutral {{ background: #854d0e; color: #fde047; }}
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>MACD Multi-Timeframe Scanner Dashboard</h1>
        <p style="color: #94a3b8; margin: 0;">Automated transition, MACD States, RSI & Divergence calculations across 4 Timeframe Pairs</p>
    </header>

    <!-- TAB NAVIGATION -->
    <div class="tab-buttons">
        <button class="tab-btn active" onclick="switchTab('tabScreener')">Stock Screener</button>
        <button class="tab-btn" onclick="switchTab('tabGuide')">Market Bias Guide (Tab 2)</button>
        <button class="tab-btn" onclick="switchTab('tabDivergence')">Divergence Scanner (Tab 3)</button>
    </div>

    <!-- TAB 1: SCREENER -->
    <div id="tabScreener" class="tab-content active">
        <div class="controls">
            <div class="control-group">
                <label for="pairSelect">1. Select Timeframe Pair (LTF -> HTF)</label>
                <select id="pairSelect" onchange="filterData()">
                    <option value="15m -> 1h">15m -> 1h</option>
                    <option value="1h -> Daily">1h -> Daily</option>
                    <option value="Daily -> Weekly">Daily -> Weekly</option>
                    <option value="Weekly -> Monthly">Weekly -> Monthly</option>
                </select>
            </div>

            <div class="control-group">
                <label for="conditionSelect">2. Select MACD Transition Condition</label>
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

        <div class="stats-badge" id="countBadge">Matching Stocks: 0</div>

        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Timeframe Pair</th>
                    <th>Close Price</th>
                    <th>HTF MACD State</th>
                    <th>LTF MACD State</th>
                    <th>HTF RSI (14)</th>
                    <th>LTF RSI (14)</th>
                    <th>Market Bias</th>
                </tr>
            </thead>
            <tbody id="stockTableBody">
            </tbody>
        </table>
    </div>

    <!-- TAB 2: MARKET BIAS GUIDE -->
    <div id="tabGuide" class="tab-content">
        <h2 style="color: var(--accent); margin-top: 0;">Market Bias & Practical Interpretation Reference</h2>
        <table>
            <thead>
                <tr>
                    <th style="width: 25%;">Market Bias</th>
                    <th style="width: 75%;">Practical Trading Interpretation</th>
                </tr>
            </thead>
            <tbody id="guideTableBody">
            </tbody>
        </table>
    </div>

    <!-- TAB 3: DIVERGENCE SCANNER -->
    <div id="tabDivergence" class="tab-content">
        <div class="controls" style="grid-template-columns: 1fr;">
            <div class="control-group">
                <label for="divPairSelect">Select Timeframe Pair</label>
                <select id="divPairSelect" onchange="filterDivergenceData()">
                    <option value="ALL">All Pairs</option>
                    <option value="15m -> 1h">15m -> 1h</option>
                    <option value="1h -> Daily">1h -> Daily</option>
                    <option value="Daily -> Weekly">Daily -> Weekly</option>
                    <option value="Weekly -> Monthly">Weekly -> Monthly</option>
                </select>
            </div>
        </div>

        <div class="summary-cards">
            <div class="card card-bullish active" id="cardBullish" onclick="setDivergenceCategory('Bullish')">
                <div class="card-title">Bullish Setups (RD/ND)</div>
                <div class="card-count" id="countBullish">0</div>
            </div>
            <div class="card card-bearish" id="cardBearish" onclick="setDivergenceCategory('Bearish')">
                <div class="card-title">Bearish Setups (RD/ND)</div>
                <div class="card-count" id="countBearish">0</div>
            </div>
            <div class="card card-neutral" id="cardNeutral" onclick="setDivergenceCategory('Neutral')">
                <div class="card-title">Neutral / No Divergence</div>
                <div class="card-count" id="countNeutral">0</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Timeframe Pair</th>
                    <th>Close Price</th>
                    <th>Setup Style (Divergence)</th>
                    <th>HTF MACD State</th>
                    <th>LTF MACD State</th>
                    <th>HTF RSI (14)</th>
                    <th>LTF RSI (14)</th>
                </tr>
            </thead>
            <tbody id="divergenceTableBody">
            </tbody>
        </table>
    </div>

</div>

<script>
    const stockData = {json_tab1};
    const guideData = {guide_json_data};
    const divergenceData = {json_tab3};

    let selectedDivCategory = 'Bullish';

    function switchTab(tabId) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        
        document.getElementById(tabId).classList.add('active');
        event.currentTarget.classList.add('active');
    }}

    function getBadgeClass(cond) {{
        if (cond.includes('Bullish') || cond.includes('Buy') || cond.includes('Recovery') || cond.includes('Accumulation')) {{
            return 'bullish';
        }} else if (cond.includes('Bearish') || cond.includes('Correction') || cond.includes('Failure')) {{
            return 'bearish';
        }}
        return 'neutral';
    }}

    function filterData() {{
        const selectedPair = document.getElementById('pairSelect').value;
        const selectedCondition = document.getElementById('conditionSelect').value;
        const tableBody = document.getElementById('stockTableBody');

        const filtered = stockData.filter(item => item.pair === selectedPair && item.condition === selectedCondition);

        document.getElementById('countBadge').innerText = `Matching Stocks: ${{filtered.length}}`;
        tableBody.innerHTML = '';

        if (filtered.length === 0) {{
            tableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #94a3b8; padding: 30px;">No stocks matching this pair & condition.</td></tr>`;
            return;
        }}

        filtered.forEach(row => {{
            const badgeClass = getBadgeClass(row.condition);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight: bold; color: var(--accent);">${{row.symbol}}</td>
                <td>${{row.pair}}</td>
                <td>${{row.close !== undefined ? row.close.toFixed(2) : 'N/A'}}</td>
                <td style="color: #cbd5e1;">${{row.htf_state}}</td>
                <td style="color: #cbd5e1;">${{row.ltf_state}}</td>
                <td style="font-weight: bold; color: ${{row.htf_rsi > 60 ? '#4ade80' : row.htf_rsi < 40 ? '#fca5a5' : '#fde047'}};">${{row.htf_rsi}}</td>
                <td style="font-weight: bold; color: ${{row.ltf_rsi > 60 ? '#4ade80' : row.ltf_rsi < 40 ? '#fca5a5' : '#fde047'}};">${{row.ltf_rsi}}</td>
                <td><span class="badge ${{badgeClass}}">${{row.condition}}</span></td>
            `;
            tableBody.appendChild(tr);
        }});
    }}

    function setDivergenceCategory(cat) {{
        selectedDivCategory = cat;
        document.querySelectorAll('.summary-cards .card').forEach(c => c.classList.remove('active'));
        document.getElementById('card' + cat).classList.add('active');
        filterDivergenceData();
    }}

    function filterDivergenceData() {{
        const selectedPair = document.getElementById('divPairSelect').value;
        const tableBody = document.getElementById('divergenceTableBody');

        let filteredByPair = divergenceData;
        if (selectedPair !== 'ALL') {{
            filteredByPair = divergenceData.filter(item => item.pair === selectedPair);
        }}

        const countBullish = filteredByPair.filter(i => i.category === 'Bullish').length;
        const countBearish = filteredByPair.filter(i => i.category === 'Bearish').length;
        const countNeutral = filteredByPair.filter(i => i.category === 'Neutral').length;

        document.getElementById('countBullish').innerText = countBullish;
        document.getElementById('countBearish').innerText = countBearish;
        document.getElementById('countNeutral').innerText = countNeutral;

        const filtered = filteredByPair.filter(item => item.category === selectedDivCategory);

        tableBody.innerHTML = '';
        if (filtered.length === 0) {{
            tableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #94a3b8; padding: 30px;">No stocks matching current filters.</td></tr>`;
            return;
        }}

        filtered.forEach(row => {{
            const badgeClass = getBadgeClass(row.divergence);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight: bold; color: var(--accent);">${{row.symbol}}</td>
                <td>${{row.pair}}</td>
                <td>${{row.close !== undefined ? row.close.toFixed(2) : 'N/A'}}</td>
                <td><span class="badge ${{badgeClass}}">${{row.divergence}}</span></td>
                <td style="color: #cbd5e1;">${{row.htf_state}}</td>
                <td style="color: #cbd5e1;">${{row.ltf_state}}</td>
                <td style="font-weight: bold; color: ${{row.htf_rsi > 60 ? '#4ade80' : row.htf_rsi < 40 ? '#fca5a5' : '#fde047'}};">${{row.htf_rsi}}</td>
                <td style="font-weight: bold; color: ${{row.ltf_rsi > 60 ? '#4ade80' : row.ltf_rsi < 40 ? '#fca5a5' : '#fde047'}};">${{row.ltf_rsi}}</td>
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
            tr.innerHTML = `
                <td><span class="badge ${{badgeClass}}">${{row['Market Bias']}}</span></td>
                <td style="color: #cbd5e1; line-height: 1.5;">${{row['Interpretation']}}</td>
            `;
            guideBody.appendChild(tr);
        }});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
        filterData();
        populateGuide();
        filterDivergenceData();
    }});
</script>

</body>
</html>"""

if __name__ == '__main__':
    tab1_data, tab3_data = process_stock_data()
    build_html_dashboard(tab1_data, tab3_data)
