import os
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Multi-Timeframe Stock Visualizer", layout="wide")
st.title("📈 Multi-Timeframe Stock Chart Visualizer")

# 1. Define base directory and timeframe directory mappings
BASE_DIR = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent / "stockdata_D").exists() else Path(__file__).resolve().parent

TIMEFRAME_FOLDERS = {
    "15-Min": BASE_DIR / "stockdata_15",
    "1-Hour": BASE_DIR / "stockdata_1H",
    "Daily": BASE_DIR / "stockdata_D",
    "Weekly": BASE_DIR / "stockdata_W",
    "Monthly": BASE_DIR / "stockdata_M",
}

# 2. Get available stock symbols across all timeframe directories
@st.cache_data
def get_available_symbols(folders):
    symbols = set()
    for tf_name, folder_path in folders.items():
        if folder_path.exists():
            for file in folder_path.glob("*.json"):
                # Extract stock symbol name (e.g., ABB.json -> ABB)
                symbols.add(file.stem.upper())
    return sorted(list(symbols))

symbols = get_available_symbols(TIMEFRAME_FOLDERS)

if not symbols:
    st.error("No JSON stock files found in the stockdata folders. Please verify your folder locations.")
    st.stop()

# 3. Dropdown Selection for Stock Symbol
selected_symbol = st.selectbox("Select Stock Symbol:", symbols)

# 4. Fetch JSON data across all available timeframe folders for the selected symbol
def load_symbol_timeframes(folders, symbol):
    timeframe_data = {}
    
    for tf_name, folder_path in folders.items():
        # Match case-insensitively for the symbol
        file_path = folder_path / f"{symbol}.json"
        if not file_path.exists():
            # Fallback search if capitalization differs
            matches = list(folder_path.glob(f"{symbol}.json")) or list(folder_path.glob(f"{symbol.lower()}.json"))
            if matches:
                file_path = matches[0]

        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    
                    # Convert list or dictionary JSON payload to DataFrame
                    if isinstance(data, dict):
                        # Handle payloads wrapped under keys like "data" or "candles"
                        data = data.get("data", data.get("candles", data))
                        
                    df = pd.DataFrame(data)
                    df.columns = [col.lower() for col in df.columns]
                    
                    # Standardize datetime column
                    time_col = next((col for col in ['datetime', 'date', 'time', 'timestamp'] if col in df.columns), None)
                    if time_col:
                        df['datetime'] = pd.to_datetime(df[time_col])
                        df = df.sort_values('datetime')
                        timeframe_data[tf_name] = df
            except Exception as e:
                st.warning(f"Error loading {tf_name} data for {symbol}: {e}")
                
    return timeframe_data

all_timeframes = load_symbol_timeframes(TIMEFRAME_FOLDERS, selected_symbol)

# 5. Display Candlestick Charts in Timeframe Tabs
if all_timeframes:
    tf_tabs = st.tabs(list(all_timeframes.keys()))
    
    for tab, (tf_name, df) in zip(tf_tabs, all_timeframes.items()):
        with tab:
            st.subheader(f"{selected_symbol} — {tf_name}")
            
            fig = go.Figure(data=[go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=f"{selected_symbol} {tf_name}"
            )])

            fig.update_layout(
                title=f"{selected_symbol} ({tf_name})",
                yaxis_title="Price",
                xaxis_title="Date / Time",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View Raw Data"):
                st.dataframe(df)
else:
    st.warning(f"No JSON data found for '{selected_symbol}' across the stockdata folders.")
