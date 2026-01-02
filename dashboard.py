import streamlit as st
import pandas as pd

st.title("⚽ Vision-Based Soccer Analyzer")

# 1. Load Data
st.header("1. Raw Player Data")
try:
    df = pd.read_csv("player_movements.csv")
    st.write("Data loaded successfully!")
    st.dataframe(df.head()) # Show first few rows
except FileNotFoundError:
    st.error("CSV file not found. Please run main.py first.")
    st.stop()

# 2. Select a Player to Analyze
player_ids = df['PlayerID'].unique()
selected_player = st.selectbox("Select Player ID to Analyze:", player_ids)

# Filter data for that single player
player_df = df[df['PlayerID'] == selected_player].copy()

# 3. "Smoothing Filter" (Resume Point!)
# Raw AI data is jittery. We calculate a 'Moving Average' to smooth the path.
st.subheader("Trajectory Smoothing")
window_size = st.slider("Smoothing Window (Frames):", min_value=1, max_value=20, value=5)

player_df['x_smooth'] = player_df['x'].rolling(window=window_size).mean()
player_df['y_smooth'] = player_df['y'].rolling(window=window_size).mean()

# 4. Visualization: Heatmap / Movement Path
st.header("2. Player Movement Path")
st.line_chart(player_df[['x_smooth', 'y_smooth']])

# 5. Speed Calculation (Simple approximation)
# Calculate distance between current frame and previous frame
player_df['dist'] = ((player_df['x_smooth'].diff())**2 + (player_df['y_smooth'].diff())**2)**0.5
# Normalize speed (just a rough score for now since we don't have meters)
player_df['speed'] = player_df['dist'].fillna(0) * 100 

st.header("3. Sprint Metrics")
st.line_chart(player_df['speed'])

st.metric(label="Top Speed (Arbitrary Units)", value=f"{player_df['speed'].max():.2f}")