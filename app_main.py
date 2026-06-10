import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from config import OPENWEATHER_API_KEY
from lstm_model import EnergyLSTM
from api_service import fetch_real_energy_prices
from tensorflow.keras.models import load_model
from api_service import fetch_weather_forecast


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Energy Trading System",
    page_icon="⚡",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(0,255,255,0.2);
}

.stButton>button {
    background: linear-gradient(to right, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
}

h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# DATABASE SETUP
# =========================

conn = sqlite3.connect("energy_trading.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS tradess (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    temperature REAL,
    market_price REAL,
    prophet_forecast REAL,
    lstm_forecast REAL,
    decision TEXT
)
''')

conn.commit()

# =========================
# HEADER
# =========================

st.title("⚡ AI-Driven Energy Trading System")
st.markdown("### Real-Time Smart Trading Intelligence Dashboard")

# =========================
# WEATHER API
# =========================

API_KEY = OPENWEATHER_API_KEY
CITY = "Bhopal"


def get_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        temp = data['main']['temp']
        weather = data['weather'][0]['main']

        return temp, weather

    except:
        return 30, "Clear"

# =========================
# PROPHET FORECASTING
# =========================

def forecast_price(df):

    prophet_df = pd.DataFrame()

    prophet_df['ds'] = pd.to_datetime(df['date'])

    prophet_df['y'] = pd.to_numeric(
        df['price'],
        errors='coerce'
    )

    prophet_df = prophet_df.dropna()

    model = Prophet()

    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=12,
        freq='ME'
    )

    forecast = model.predict(future)

    predicted_price = forecast['yhat'].iloc[-1]

    return predicted_price, forecast

# =========================
# LOAD DATA
# =========================

energy_df = fetch_real_energy_prices()
energy_df = energy_df.rename(columns={
    'period': 'date',
    'price': 'price'
})

energy_df['date'] = pd.to_datetime(
    energy_df['date']
)

predicted_price, forecast = forecast_price(energy_df)

# =========================
# TRAIN LSTM
# =========================
# lstm = EnergyLSTM(
#     "energy_dataset.csv"
# )

# lstm.train()

# lstm.model.save(
#     "lstm_energy.keras"
# )

lstm_engine = EnergyLSTM(
    "energy_dataset.csv"
)

lstm_engine.model = load_model(
    "lstm_energy.keras"
)

future_lstm_price = (
    lstm_engine.predict_next_price()
)

# =========================
# LIVE WEATHER
# =========================

current_temp, weather_condition = get_weather()

# =========================
# FETCH DATA FROM API
# =========================

weather_forecast = fetch_weather_forecast()

weather_df = pd.DataFrame(weather_forecast)

# =========================
# WEATHER FORECAST GRAPH
# =========================

st.subheader("🌦️ Weather Forecast Analysis")

weather_fig = px.line(
    weather_df,
    x='datetime',
    y='temperature',
    title='Upcoming Temperature Forecast',
    markers=True
)

weather_fig.update_layout(
    xaxis_title='Date & Time',
    yaxis_title='Temperature (°C)'
)

st.plotly_chart(weather_fig, use_container_width=True)

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("⚙️ Trading Inputs")

forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [7, 14, 30]
)


current_demand = st.sidebar.slider("Energy Demand", 0, 150, 75)
current_price = st.sidebar.slider("Current Market Price", 0, 150, 60)

# Weather effect simulation
if current_temp > 35:
    current_demand += 10


# =========================
# AI DECISION ENGINE
# =========================

current_price = float(
    energy_df['price'].iloc[-1]
)

delta = (
    future_lstm_price -
    current_price
)

def trading_signal(
    current_price,
    predicted_price
):

    difference = predicted_price - current_price

    confidence = abs(difference)

    if difference > 5:
        return "BUY", "#16a34a", confidence

    elif difference < -5:
        return "SELL", "#dc2626", confidence

    else:
        return "HOLD", "#f59e0b", confidence


decision, color, confidence = trading_signal(
    current_price,
    future_lstm_price
)

st.metric(
    "🎯 AI Confidence",
    f"{confidence:.2f}"
)

difference = future_lstm_price - current_price

st.info(
    f"""
    Current Price: {current_price:.2f}

    Predicted Price: {future_lstm_price:.2f}

    Difference: {difference:.2f}

    Recommendation: {decision}
    """
)

# =========================
# SAVE TO DATABASE
# =========================

cursor.execute(
    '''
    INSERT INTO tradess
    (
        timestamp,
        temperature,
        market_price,
        prophet_forecast,
        lstm_forecast,
        decision
    )
    VALUES (?, ?, ?, ?, ?, ?)
    ''',
    (
        str(datetime.now()),
        current_temp,
        current_price,
        float(predicted_price),
        float(future_lstm_price),
        decision
    )
)

conn.commit()

# =========================
# METRICS SECTION (REQUIRES CHANGES)
# =========================

col1,col2,col3,col4,col5 = st.columns(5)

col1.metric(
    "🌡 Temperature",
    f"{current_temp:.1f}°C"
)

col2.metric(
    "⚡ Market Price",
    f"{current_price:.2f}"
)

col3.metric(
    "🔮 Prophet",
    f"{predicted_price:.2f}"
)

col4.metric(
    "🧠 LSTM",
    f"{future_lstm_price:.2f}",
    delta=f"{future_lstm_price-current_price:.2f}"
)

col5.metric(
    "🚦 Signal",
    decision
)

# =========================
# AI SIGNAL BOX
# =========================

st.markdown("---")

st.markdown(
    f"""
    <div style='background-color:{color}; padding:25px; border-radius:15px;'>
    <h1 style='text-align:center; color:white;'>AI Recommendation: {decision}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# PRICE TREND GRAPH
# =========================

st.subheader("📈 Energy Price Trend")

fig = px.line(
    energy_df,
    x='date',
    y='price',
    title='Historical Energy Prices'
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# FORECAST GRAPH
# =========================

st.subheader("🔮 AI Forecast")

forecast_fig = go.Figure()

forecast_fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast'
))

st.plotly_chart(forecast_fig, use_container_width=True)

# =========================
# LSTM GRAPH
# =========================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=lstm_engine.df["price actual"][-200:],
        mode="lines",
        name="Historical"
    )
)

fig.add_trace(
    go.Scatter(
        x=[200],
        y=[future_lstm_price],
        mode="markers",
        marker=dict(
            size=15
        ),
        name="Prediction"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================
# DATABASE HISTORY
# =========================

st.subheader("📜 Trading History")

history = pd.read_sql_query(
    "SELECT * FROM tradess ORDER BY id DESC LIMIT 10",
    conn
)

st.dataframe(history, use_container_width=True)

# =========================
# Price Distribution
# =========================

# st.subheader("📊 Price Distribution")

# hist = px.histogram(
#     energy_df,
#     x="price",
#     nbins=25
# )

# st.plotly_chart(
#     hist,
#     use_container_width=True
# )

# =========================
# Prophet vs LSTM
# =========================

# comparison = pd.DataFrame({
#     "Model": [
#         "Prophet",
#         "LSTM"
#     ],
#     "Forecast": [
#         predicted_price,
#         future_lstm_price
#     ]
# })

# comparison_fig = px.bar(
#     comparison,
#     x="Model",
#     y="Forecast"
# )

# st.plotly_chart(
#     comparison_fig,
#     use_container_width=True
# )

# =========================
# Rolling Average
# =========================

# energy_df["rolling_avg"] = (
#     energy_df["price"]
#     .rolling(12)
#     .mean()
# )

# rolling_fig = px.line(
#     energy_df,
#     x="date",
#     y=["price", "rolling_avg"]
# )

# st.plotly_chart(
#     rolling_fig,
#     use_container_width=True
# )

# =========================
# FOOTER
# =========================

st.markdown("---")

