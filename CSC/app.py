import os
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Multi-Timeframe Stock Visualizer", layout="wide")
st.title("📈 Multi-Timeframe Stock Chart Visualizer")

# 1. Multi-level fallback to accurately discover repo root
def locate_repo_root():
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent,           # /mount/src/data-collector/CSC
        current_file.parent.parent,    # /mount/src/data-collector (Repo Root)
        Path.cwd(),                    # Current working directory
        Path.cwd().parent              # Parent of CWD
    ]
    
    for candidate in candidates:
        if (candidate / "stockdata_D").exists():
            return candidate
            
    # Default to parent.parent if folder isn't found yet
    return current_file.parent.parent

BASE_DIR = locate_repo_root()

TIMEFRAME_FOLDERS = {
    "15-Min": BASE_DIR / "stockdata_15",
    "1-Hour": BASE_DIR / "stockdata_1H",
    "Daily": BASE_DIR / "stockdata_D",
    "Weekly": BASE_DIR / "stockdata_W",
    "Monthly": BASE_DIR / "stockdata_M",
}

# Debug section (expandable to inspect active directory paths)
with st.sidebar.expander("🔍 Directory Diagnostics"):
    st.write(f"**Repo Root Detected:** `{BASE_DIR}`")
    for name, path in TIMEFRAME_FOLDERS.items():
        st.write(f"**{name}:** `{path}` (Exists: `{path.exists()}`)")

# 2. Extract stock symbols from all valid timeframe directories
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
    st.error(f"No JSON stock files found in `{BASE_DIR}`. Open 'Directory Diagnostics' in the sidebar to check folder path locations.")
    st.stop()

# 3. Dropdown selection for stock symbol
selected_symbol = st.selectbox("Select Stock Symbol:", symbols)

# 4. Load timeframe datasets for selected symbol
def load_symbol_timeframes(folders, symbol):
    timeframe_data = {}
    
    for tf_name, folder_path in folders.items():
        if not folder_path.exists():
            continue
            
        # Match case-insensitively for JSON files
        matches = list(folder_path.glob(f"{symbol}.json")) or list(folder_path.glob(f"{symbol.lower()}.json"))
        
        if matches:
            file_path = matches[0]
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    
                    if isinstance(data, dict):
                        data = data.get("data", data.get("candles", data))
                        
                    df = pd.DataFrame(data)
                    df.columns = [col.lower() for col in df.columns]
                    
                    time_col = next((col for col in ['datetime', 'date', 'time', 'timestamp'] if col in df.columns), None)
                    if time_col:
                        df['datetime'] = pd.to_datetime(df[time_col])
                        df = df.sort_values('datetime')
                        timeframe_data[tf_name] = df
            except Exception as e:
                st.warning(f"Error loading {tf_name} data for {symbol}: {e}")
                
    return timeframe_data

all_timeframes = load_symbol_timeframes(TIMEFRAME_FOLDERS, selected_symbol)

# 5. Render charts
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
    st.warning(f"No JSON data found for symbol '{selected_symbol}'.")
