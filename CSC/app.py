import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Multi-Timeframe Stock Visualizer", layout="wide")
st.title("📈 Multi-Timeframe Quadrant Visualizer (EMA, RSI, MACD & Divergence)")

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

# 3. Technical Indicator & Divergence Calculations
def compute_indicators(df):
    if len(df) < 50:
        return df
    
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

    # Simple Divergence Detection (RSI)
    df['bullish_div'] = False
    df['bearish_div'] = False
    
    for i in range(14, len(df)):
        # Regular Bullish Divergence: Price making Lower Lows, RSI making Higher Lows
        if df['close'].iloc[i] < df['close'].iloc[i-5] and df['rsi'].iloc[i] > df['rsi'].iloc[i-5]:
            if df['rsi'].iloc[i] < 40: # Near oversold condition
                df.iloc[i, df.columns.get_loc('bullish_div')] = True
                
        # Regular Bearish Divergence: Price making Higher Highs, RSI making Lower Highs
        if df['close'].iloc[i] > df['close'].iloc[i-5] and df['rsi'].iloc[i] < df['rsi'].iloc[i-5]:
            if df['rsi'].iloc[i] > 60: # Near overbought condition
                df.iloc[i, df.columns.get_loc('bearish_div')] = True
                
    return df

# 4. Load Data across 4 Timeframes
def load_quadrant_data(folders, symbol):
    timeframe_data = {}
    for tf_name in ["1-Hour", "Daily", "Weekly", "Monthly"]:
        folder_path = folders.get(tf_name)
        if folder_path and folder_path.exists():
            matches = list(folder_path.glob(f"{symbol}.json")) or list(folder_path.glob(f"{symbol.lower()}.json"))
            if matches:
                try:
                    with open(matches[0], "r") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        data = data.get("data", data.get("candles", data))
                    df = pd.DataFrame(data)
                    df.columns = [col.lower() for col in df.columns]
                    time_col = next((c for c in ['datetime', 'date', 'time', 'timestamp'] if c in df.columns), None)
                    if time_col:
                        df['datetime'] = pd.to_datetime(df[time_col])
                        df = df.sort_values('datetime').reset_index(drop=True)
                        timeframe_data[tf_name] = compute_indicators(df)
                except Exception as e:
                    st.warning(f"Error loading {tf_name} for {symbol}: {e}")
    return timeframe_data

quad_data = load_quadrant_data(TIMEFRAME_FOLDERS, selected_symbol)

# 5. Render 4-Panel Quadrant Subplots
if quad_data:
    # 2x2 Grid Layout Specs (Each Quadrant has 3 sub-rows: Price, RSI, MACD)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("1-Hour Timeframe", "Daily Timeframe", "Weekly Timeframe", "Monthly Timeframe"),
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    positions = {
        "1-Hour": (1, 1),
        "Daily": (1, 2),
        "Weekly": (2, 1),
        "Monthly": (2, 2)
    }

    for tf_name, (row, col) in positions.items():
        if tf_name in quad_data:
            df = quad_data[tf_name]
            
            # Candlestick Chart
            fig.add_trace(go.Candlestick(
                x=df['datetime'], open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name=f"{tf_name} Price",
                showlegend=False
            ), row=row, col=col)
            
            # EMA 20
            if 'ema_20' in df:
                fig.add_trace(go.Scatter(
                    x=df['datetime'], y=df['ema_20'],
                    line=dict(color='yellow', width=1), name="EMA 20", showlegend=False
                ), row=row, col=col)
                
            # EMA 50
            if 'ema_50' in df:
                fig.add_trace(go.Scatter(
                    x=df['datetime'], y=df['ema_50'],
                    line=dict(color='cyan', width=1), name="EMA 50", showlegend=False
                ), row=row, col=col)
                
            # Divergence Markers
            if 'bullish_div' in df and df['bullish_div'].any():
                bull_df = df[df['bullish_div']]
                fig.add_trace(go.Scatter(
                    x=bull_df['datetime'], y=bull_df['low'] * 0.98,
                    mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='Bullish Div', showlegend=False
                ), row=row, col=col)

            if 'bearish_div' in df and df['bearish_div'].any():
                bear_df = df[df['bearish_div']]
                fig.add_trace(go.Scatter(
                    x=bear_df['datetime'], y=bear_df['high'] * 1.02,
                    mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='Bearish Div', showlegend=False
                ), row=row, col=col)

    fig.update_layout(
        title=f"Quadrant Technical Analysis: {selected_symbol}",
        template="plotly_dark",
        height=900,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        xaxis4_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Legend Reference
    st.info("💡 **Chart Indicators Key:** 🟨 **EMA 20** | 🟦 **EMA 50** | 🟢 **Bullish Divergence (Triangle Up)** | 🔴 **Bearish Divergence (Triangle Down)**")
else:
    st.warning("No quadrant timeframe data found for selected stock.")
