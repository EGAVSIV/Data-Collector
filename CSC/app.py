import os
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. Set the directory path where your JSON files reside
DATA_DIR = "./Data"

st.set_page_config(page_title="Stock Timeframe Dashboard", layout="wide")
st.title("📈 Multi-Timeframe Stock Chart Visualizer")

# 2. Get available stock symbols from the Data directory
def get_available_symbols(directory):
    if not os.path.exists(directory):
        return []
    
    files = os.listdir(directory)
    # Extract unique symbol names (assuming filenames like ABB_1D.json, ABB_1H.json, or ABB.json)
    symbols = set()
    for f in files:
        if f.endswith('.json'):
            # Strip extension and split timeframe if present (e.g., ABB_1D -> ABB)
            symbol_name = f.replace('.json', '').split('_')[0]
            symbols.add(symbol_name)
            
    return sorted(list(symbols))

symbols = get_available_symbols(DATA_DIR)

if not symbols:
    st.error(f"No JSON stock files found in '{DATA_DIR}' directory. Please verify your folder location.")
    st.stop()

# 3. Dropdown Selection for Stock Symbol
selected_symbol = st.selectbox("Select Stock Symbol:", symbols)

# 4. Find all timeframe JSON files related to the selected symbol
def load_symbol_timeframes(directory, symbol):
    timeframe_data = {}
    
    for filename in os.listdir(directory):
        if filename.startswith(symbol) and filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Extract timeframe label (e.g., ABB_1D.json -> 1D, ABB.json -> Default)
            parts = filename.replace('.json', '').split('_')
            timeframe_label = parts[1] if len(parts) > 1 else "Default"
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Convert JSON to pandas DataFrame
                    df = pd.DataFrame(data)
                    
                    # Standardize column names (lowercase)
                    df.columns = [col.lower() for col in df.columns]
                    
                    # Ensure datetime formatting
                    if 'date' in df.columns:
                        df['datetime'] = pd.to_datetime(df['date'])
                    elif 'time' in df.columns:
                        df['datetime'] = pd.to_datetime(df['time'])
                    elif 'timestamp' in df.columns:
                        df['datetime'] = pd.to_datetime(df['timestamp'])
                        
                    df = df.sort_values('datetime')
                    timeframe_data[timeframe_label] = df
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
                
    return timeframe_data

# Fetch data for selected stock symbol
all_timeframes = load_symbol_timeframes(DATA_DIR, selected_symbol)

# 5. Display Charts for Each Timeframe
if all_timeframes:
    # Allow user to pick which timeframe to view, or show all
    tf_tabs = st.tabs(list(all_timeframes.keys()))
    
    for tab, (tf_name, df) in zip(tf_tabs, all_timeframes.items()):
        with tab:
            st.subheader(f"{selected_symbol} — {tf_name} Timeframe")
            
            # Plot Interactive Candlestick Chart
            fig = go.Figure(data=[go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=f"{selected_symbol} {tf_name}"
            )])

            fig.update_layout(
                title=f"{selected_symbol} Price Chart ({tf_name})",
                yaxis_title="Stock Price",
                xaxis_title="Date / Time",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Show Raw Data Table optionally
            with st.expander("View Raw Data"):
                st.dataframe(df)
else:
    st.warning(f"No valid JSON data found for symbol '{selected_symbol}'.")
