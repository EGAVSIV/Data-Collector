import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="Multi-Timeframe Quadrant Analysis", layout="wide")
st.title("📈 Advanced Multi-Timeframe Quadrant Visualizer")

# 1. Directory Locator
def locate_repo_root():
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent,
        current_file.parent.parent,
        Path.cwd(),
        Path.cwd().parent
    ]
    for candidate in candidates:
        if (candidate / "stockdata_D").exists():
            return candidate
    return current_file.parent.parent

BASE_DIR = locate_repo_root()

TIMEFRAME_FOLDERS = {
    "1-Hour": BASE_DIR / "stockdata_1H",
    "Daily": BASE_DIR / "stockdata_D",
    "Weekly": BASE_DIR / "stockdata_W",
    "Monthly": BASE_DIR / "stockdata_M",
}

# 2. Extract Stock Symbols
@st.cache_data
def get_available_symbols(folders):
    symbols = set()
    for tf_name, folder_path in folders.items():
        if folder_path.exists():
            for file in folder_path.glob("*.json"):
                symbols.add(file.stem.upper())
    return sorted(list(symbols))

symbols = get_available_symbols(TIMEFRAME_FOLDERS)

if not symbols:
    st.error(f"No JSON stock files found in `{BASE_DIR}`.")
    st.stop()

selected_symbol = st.selectbox("Select Stock Symbol:", symbols)

# 3. Technical Indicators & Normal/Hidden Divergence Detection
def compute_indicators_and_divergences(df):
    if len(df) < 30:
        return df, []
    
    # Calculate EMAs
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Divergence Identification Logic (Pivot points comparison)
    divergences = []
    window = 5
    
    for i in range(window * 2, len(df) - window):
        p1, p2 = i - window, i
        
        # Price Lows & RSI Lows
        price_low_1, price_low_2 = df['low'].iloc[p1], df['low'].iloc[p2]
        rsi_low_1, rsi_low_2 = df['rsi'].iloc[p1], df['rsi'].iloc[p2]
        
        # Price Highs & RSI Highs
        price_high_1, price_high_2 = df['high'].iloc[p1], df['high'].iloc[p2]
        rsi_high_1, rsi_high_2 = df['rsi'].iloc[p1], df['rsi'].iloc[p2]
        
        # 1. Normal Bullish Divergence (Lower Low in Price, Higher Low in RSI)
        if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1:
            divergences.append({"type": "Normal Bullish", "p1": p1, "p2": p2, "color": "#00FF00"})
            
        # 2. Hidden Bullish Divergence (Higher Low in Price, Lower Low in RSI)
        elif price_low_2 > price_low_1 and rsi_low_2 < rsi_low_1:
            divergences.append({"type": "Hidden Bullish", "p1": p1, "p2": p2, "color": "#32CD32"})
            
        # 3. Normal Bearish Divergence (Higher High in Price, Lower High in RSI)
        if price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1:
            divergences.append({"type": "Normal Bearish", "p1": p1, "p2": p2, "color": "#FF0000"})
            
        # 4. Hidden Bearish Divergence (Lower High in Price, Higher High in RSI)
        elif price_high_2 < price_high_1 and rsi_high_2 > rsi_high_1:
            divergences.append({"type": "Hidden Bearish", "p1": p1, "p2": p2, "color": "#FF6347"})
            
    return df, divergences

# 4. Data Fetcher
def load_data(folders, symbol):
    data_dict = {}
    for tf_name in ["1-Hour", "Daily", "Weekly", "Monthly"]:
        folder_path = folders.get(tf_name)
        if folder_path and folder_path.exists():
            matches = list(folder_path.glob(f"{symbol}.json")) or list(folder_path.glob(f"{symbol.lower()}.json"))
            if matches:
                try:
                    with open(matches[0], "r") as f:
                        raw = json.load(f)
                    if isinstance(raw, dict):
                        raw = raw.get("data", raw.get("candles", raw))
                    df = pd.DataFrame(raw)
                    df.columns = [c.lower() for c in df.columns]
                    time_col = next((c for c in ['datetime', 'date', 'time', 'timestamp'] if c in df.columns), None)
                    if time_col:
                        df['datetime'] = pd.to_datetime(df[time_col])
                        df = df.sort_values('datetime').reset_index(drop=True)
                        df, divs = compute_indicators_and_divergences(df)
                        data_dict[tf_name] = (df, divs)
                except Exception as e:
                    st.warning(f"Error loading {tf_name} for {symbol}: {e}")
    return data_dict

data_map = load_data(TIMEFRAME_FOLDERS, selected_symbol)

# 5. Build Quadrant Grid with Individual Subplots for Price, RSI, MACD
if data_map:
    # 2x2 Quadrant structure where each Quadrant contains 3 stacked sub-plots:
    # Subplots layout: 6 rows total, 2 columns
    fig = make_subplots(
        rows=6, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.03,
        horizontal_spacing=0.06,
        row_heights=[0.22, 0.08, 0.08, 0.22, 0.08, 0.08],
        subplot_titles=(
            f"{selected_symbol} - 1-Hour Price", f"{selected_symbol} - Daily Price",
            "", "", "", "",
            f"{selected_symbol} - Weekly Price", f"{selected_symbol} - Monthly Price",
            "", "", "", ""
        )
    )

    quad_positions = {
        "1-Hour":  {"price": 1, "rsi": 2, "macd": 3, "col": 1},
        "Daily":   {"price": 1, "rsi": 2, "macd": 3, "col": 2},
        "Weekly":  {"price": 4, "rsi": 5, "macd": 6, "col": 1},
        "Monthly": {"price": 4, "rsi": 5, "macd": 6, "col": 2},
    }

    for tf_name, cfg in quad_positions.items():
        if tf_name in data_map:
            df, divs = data_map[tf_name]
            col = cfg["col"]
            p_row, r_row, m_row = cfg["price"], cfg["rsi"], cfg["macd"]

            # --- Price Chart ---
            fig.add_trace(go.Candlestick(
                x=df['datetime'], open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name=f"{tf_name} Price", showlegend=False
            ), row=p_row, col=col)

            # EMAs
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['ema_20'], line=dict(color='yellow', width=1), name="EMA 20", showlegend=False), row=p_row, col=col)
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['ema_50'], line=dict(color='cyan', width=1), name="EMA 50", showlegend=False), row=p_row, col=col)

            # --- RSI Subchart ---
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['rsi'], line=dict(color='purple', width=1.5), name="RSI 14", showlegend=False), row=r_row, col=col)
            # RSI Overbought/Oversold thresholds
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=r_row, col=col)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=r_row, col=col)

            # --- MACD Subchart ---
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['macd'], line=dict(color='blue', width=1.2), name="MACD", showlegend=False), row=m_row, col=col)
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['macd_signal'], line=dict(color='orange', width=1.2), name="Signal", showlegend=False), row=m_row, col=col)
            
            # Color MACD Histogram
            colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
            fig.add_trace(go.Bar(x=df['datetime'], y=df['macd_hist'], marker_color=colors, name="Hist", showlegend=False), row=m_row, col=col)

            # --- Divergence Trendlines (Connecting Price & RSI) ---
            for div in divs[-8:]: # Display recent detected divergences
                p1, p2 = div["p1"], div["p2"]
                c = div["color"]
                
                # Draw line on Price Chart
                fig.add_trace(go.Scatter(
                    x=[df['datetime'].iloc[p1], df['datetime'].iloc[p2]],
                    y=[df['close'].iloc[p1], df['close'].iloc[p2]],
                    mode='lines+markers', line=dict(color=c, width=2, dash='dot'), showlegend=False
                ), row=p_row, col=col)

                # Draw line on RSI Subchart
                fig.add_trace(go.Scatter(
                    x=[df['datetime'].iloc[p1], df['datetime'].iloc[p2]],
                    y=[df['rsi'].iloc[p1], df['rsi'].iloc[p2]],
                    mode='lines+markers', line=dict(color=c, width=2, dash='dot'), showlegend=False
                ), row=r_row, col=col)

    fig.update_layout(
        template="plotly_dark",
        height=1400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
    )

    # Disable rangesliders across all subplots
    for i in range(1, 13):
        fig.update_xaxes(rangeslider_visible=False, row=(i-1)//2 + 1, col=(i-1)%2 + 1)

    # 6. Render as Interactive HTML to enable full Zoom/Pan toolbar controls
    html_content = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
            "displaylogo": False
        }
    )

    st.subheader("Interactive Quadrant Dashboard (Zoom In / Zoom Out Controls Enabled)")
    st.info("💡 **Divergence Line Key:** 🟩 **Normal Bullish** | 🟢 **Hidden Bullish** | 🟥 **Normal Bearish** | 🔴 **Hidden Bearish**")
    
    # Display HTML Plotly canvas
    components.html(html_content, height=1450, scrolling=True)

else:
    st.warning("No data found for selected symbol.")
