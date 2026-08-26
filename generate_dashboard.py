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

# --- Technical Indicator Functions ---
def calculate_macd(df, fast=12, slow=26, signal=9):
    if df.empty or len(df) < slow + signal:
        return None, None
    
    # Ensure dataframe is sorted by timestamp/date
    if 'datetime' in df.columns:
        df = df.sort_values('datetime')
    elif 'date' in df.columns:
        df = df.sort_values('date')

    close = df['close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line, signal_line

def get_macd_state(macd_val, signal_val):
    if pd.isna(macd_val) or pd.isna(signal_val):
        return None
    is_pco = macd_val > signal_val
    is_above_zero = macd_val > 0
    
    if is_pco and is_above_zero:
        return 1  # PCO > 0
    elif is_pco and not is_above_zero:
        return 2  # PCO < 0
    elif not is_pco and is_above_zero:
        return 3  # NCO > 0
    else:
        return 4  # NCO < 0

def map_transition(ltf_prev, ltf_curr, htf_curr):
    if None in (ltf_prev, ltf_curr, htf_curr):
        return None
    
    # 16 Condition Mapping Logic
    if htf_curr == 1: # HTF: PCO > 0
        if ltf_prev == 3 and ltf_curr == 1: return "Strong Bullish"
        if ltf_prev == 4 and ltf_curr == 2: return "Bullish Reversal"
        if ltf_prev == 1 and ltf_curr == 3: return "Bullish Pullback"
        if ltf_prev == 2 and ltf_curr == 4: return "Deep Pullback"
        
    elif htf_curr == 2: # HTF: PCO < 0
        if ltf_prev == 3 and ltf_curr == 1: return "Dip Buy / Reversal"
        if ltf_prev == 4 and ltf_curr == 2: return "Oversold Accumulation"
        if ltf_prev == 1 and ltf_curr == 3: return "Neutral / Consolidating"
        if ltf_prev == 2 and ltf_curr == 4: return "Complex Reversal Failure"
        
    elif htf_curr == 3: # HTF: NCO > 0
        if ltf_prev == 3 and ltf_curr == 1: return "Counter-Trend Buy"
        if ltf_prev == 4 and ltf_curr == 2: return "Early Recovery Attempt"
        if ltf_prev == 1 and ltf_curr == 3: return "Correction in Progress"
        if ltf_prev == 2 and ltf_curr == 4: return "Accelerating Correction"
        
    elif htf_curr == 4: # HTF: NCO < 0
        if ltf_prev == 3 and ltf_curr == 1: return "Aggressive Counter-Trend"
        if ltf_prev == 4 and ltf_curr == 2: return "Weak Oversold Bounce"
        if ltf_prev == 1 and ltf_curr == 3: return "Bearish Continuation"
        if ltf_prev == 2 and ltf_curr == 4: return "Strong Bearish"

    return None

# --- Main Data Processing Pipeline ---
def process_stock_data():
    results = []
    
    # List all unique symbols across JSON files
    sample_folder = TF_FOLDERS['D']
    if not os.path.exists(sample_folder):
        print(f"Directory {sample_folder} not found. Please check paths.")
        return []

    symbols = [f.replace('.json', '') for f in os.listdir(sample_folder) if f.endswith('.json')]
    
    for symbol in symbols:
        tf_data = {}
        
        # Load JSON data for each timeframe
        for tf, folder in TF_FOLDERS.items():
            file_path = os.path.join(folder, f"{symbol}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        raw_data = json.load(f)
                        df = pd.DataFrame(raw_data)
                        macd, sig = calculate_macd(df)
                        if macd is not None and len(macd) >= 2:
                            prev_state = get_macd_state(macd.iloc[-2], sig.iloc[-2])
                            curr_state = get_macd_state(macd.iloc[-1], sig.iloc[-1])
                            tf_data[tf] = {
                                'prev': prev_state,
                                'curr': curr_state,
                                'close': df['close'].iloc[-1]
                            }
                except Exception as e:
                    continue
        
        # Process TF Pairs
        for ltf_key, htf_key, pair_label in TF_PAIRS:
            if ltf_key in tf_data and htf_key in tf_data:
                ltf_info = tf_data[ltf_key]
                htf_info = tf_data[htf_key]
                
                condition = map_transition(ltf_info['prev'], ltf_info['curr'], htf_info['curr'])
                if condition:
                    results.append({
                        'symbol': symbol,
                        'pair': pair_label,
                        'condition': condition,
                        'close': ltf_info['close']
                    })
                    
    return results

# --- Generate Embedded HTML Dashboard ---
def build_html_dashboard(results):
    json_data = json.dumps(results)
    
    html_content = f"""<!DOCTYPE html>
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
            max-width: 1200px;
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
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 20px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}

        th, td {{
            padding: 14px 18px;
            text-align: left;
        }}

        th {{
            background-color: #334155;
            color: var(--accent);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 13px;
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
        <p style="color: #94a3b8; margin: 0;">Automated transition & condition filter across 4 Timeframe Pairs</p>
    </header>

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
                <th>Condition / Interpretation</th>
                <th>Last Close Price</th>
            </tr>
        </thead>
        <tbody id="stockTableBody">
        </tbody>
    </table>
</div>

<script>
    const stockData = {json_data};

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
            tableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: #94a3b8; padding: 30px;">No stocks matching this pair & condition.</td></tr>`;
            return;
        }}

        filtered.forEach(row => {{
            const badgeClass = getBadgeClass(row.condition);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight: bold; color: var(--accent);">${{row.symbol}}</td>
                <td>${{row.pair}}</td>
                <td><span class="badge ${{badgeClass}}">${{row.condition}}</span></td>
                <td>${{row.close !== undefined ? row.close.toFixed(2) : 'N/A'}}</td>
            `;
            tableBody.appendChild(tr);
        }});
    }}

    // Initial load
    document.addEventListener('DOMContentLoaded', filterData);
</script>

</body>
</html>
"""
    with open('index.html', 'w') as f:
        f.write(html_content)
    print("Successfully generated index.html dashboard!")

if __name__ == '__main__':
    data = process_stock_data()
    build_html_dashboard(data)
