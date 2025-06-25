# coldchain_dashboard/app.py (SQLite fallback version)

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from datetime import datetime
from prophet import Prophet
import streamlit_authenticator as stauth
from sqlalchemy import create_engine

# --- Use SQLite instead of PostgreSQL for simplicity ---
engine = create_engine('sqlite:///coldchain.db')
conn = engine.connect()

conn.execute('''
    CREATE TABLE IF NOT EXISTS sensor_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id TEXT,
        timestamp TEXT,
        temperature REAL,
        humidity REAL,
        location TEXT
    )
''')

# --- Authentication Setup ---
names = ['Admin']
usernames = ['admin']
passwords = ['123']  # Use hashed passwords in production!
hashed_passwords = stauth.Hasher(passwords).generate()
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    'coldchain_dashboard', 'abcdef', cookie_expiry_days=1)
name, authentication_status, username = authenticator.login('Login', 'main')

if not authentication_status:
    if authentication_status is False:
        st.error("Login failed")
    else:
        st.warning("Please login to access the dashboard.")
    st.stop()

# --- App Config ---
st.set_page_config(page_title="Cold Chain Dashboard", layout="wide")
authenticator.logout('Logout', 'sidebar')
st.sidebar.success(f"Welcome {name}")

st.title("🚚 Cold Chain Monitoring Dashboard")

# --- Upload Section ---
st.header("📤 Upload Sensor Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if all(col in df.columns for col in ["vehicle_id", "timestamp", "temperature", "humidity", "location"]):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.to_sql("sensor_data", engine, if_exists="append", index=False)
        st.success("✅ Data uploaded and stored successfully!")
    else:
        st.error("⚠️ CSV must contain: vehicle_id, timestamp, temperature, humidity, location")

# --- Data Analysis Section ---
st.header("📊 Data Analytics")
data = pd.read_sql("SELECT * FROM sensor_data", conn, parse_dates=["timestamp"])

if data.empty:
    st.info("No data available. Upload a CSV to begin.")
    st.stop()

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Avg. Temperature", f"{data['temperature'].mean():.2f} °C")

with col2:
    excursions = data[data['temperature'] > 8]
    st.metric("🚨 Excursions (>8°C)", len(excursions))

with col3:
    st.metric("Total Records", len(data))

# --- Filters ---
st.subheader("📅 Filter Data")
vehicle = st.selectbox("Select Vehicle ID", options=["All"] + sorted(data['vehicle_id'].unique().tolist()))
date_range = st.date_input("Select Date Range", [])

filtered_data = data.copy()
if vehicle != "All":
    filtered_data = filtered_data[filtered_data['vehicle_id'] == vehicle]
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_data = filtered_data[(filtered_data['timestamp'] >= start_date) & 
                                  (filtered_data['timestamp'] <= end_date)]

# --- Visualizations ---
st.subheader("📈 Temperature Trends")
fig = px.line(filtered_data, x='timestamp', y='temperature', color='vehicle_id', title="Temperature Over Time")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🗺️ Location of Excursions")
excursions_map = filtered_data[filtered_data['temperature'] > 8]
if not excursions_map.empty:
    st.map(excursions_map.rename(columns={"location": "address"}))
else:
    st.info("No excursions found in selected range.")

# --- Download Report ---
st.download_button("📥 Download Filtered Data as CSV", data=filtered_data.to_csv(index=False).encode('utf-8'),
                   file_name="filtered_sensor_data.csv", mime="text/csv")

# --- Forecasting with Prophet ---
st.subheader("🔮 Forecasted Temperature")
if not filtered_data.empty:
    df_prophet = filtered_data[['timestamp', 'temperature']].rename(columns={'timestamp': 'ds', 'temperature': 'y'})
    try:
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=24, freq='H')
        forecast = model.predict(future)
        fig2 = px.line(forecast, x='ds', y='yhat', title="24-Hour Temperature Forecast")
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")
