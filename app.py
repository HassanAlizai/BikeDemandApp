import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="BikeShare Demand Predictor", layout="wide")

# ---------------- Load model (cached) ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("bike_pipeline.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

model, model_columns = load_artifacts()

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #2b7de9;
    text-align: center;
}
.subtitle {
    font-size: 20px;
    color: #555;
    text-align: center;
    margin-bottom: 20px;
}
.author {
    text-align: center;
    font-size: 16px;
    margin-bottom: 30px;
}
.predict-btn button {
    background-color: #ff4b4b !important;
    color: white !important;
    width: 100%;
    height: 55px;
    font-size: 20px;
    border-radius: 12px;
}
.result-box {
    background-color: #eaf7ea;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    color: #2e7d32;
}
.status-box {
    background-color: #e6f6ea;
    padding: 15px;
    border-radius: 10px;
    font-size: 16px;
    color: #1b5e20;
    margin-top: 10px;
}
.small-note {
    font-size: 13px;
    color: #666;
    text-align: center;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar Inputs ----------------
st.sidebar.title("Enter Conditions")

# Time controls
yr = st.sidebar.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour", 0, 23, 8)

day_name = st.sidebar.selectbox("Weekday", ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
holiday = st.sidebar.selectbox("Holiday", [0, 1])

# Weather controls (we'll convert real units -> normalized like hour.csv)
temp_c = st.sidebar.slider("Temperature (°C)", 0, 40, 28)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
windspeed_kmh = st.sidebar.slider("Wind Speed (km/h)", 0, 67, 10)

season_name = st.sidebar.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
weather_name = st.sidebar.selectbox("Weather", ["Clear", "Mist", "Light Rain/Snow", "Heavy Rain/Snow"])

st.sidebar.markdown("---")
debug = st.sidebar.checkbox("Show debug info (columns + input)", value=False)

# ---------------- Mapping ----------------
weekday_map = {"Sun": 0, "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6}
season_map = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
weather_map = {"Clear": 1, "Mist": 2, "Light Rain/Snow": 3, "Heavy Rain/Snow": 4}

weekday = int(weekday_map[day_name])
season = int(season_map[season_name])
weathersit = int(weather_map[weather_name])
holiday = int(holiday)
yr = int(yr)
mnth = int(mnth)
hr = int(hr)

# Workingday: weekday (Mon-Fri) AND not holiday
workingday = 1 if (weekday in [1, 2, 3, 4, 5] and holiday == 0) else 0

# Normalize inputs to match hour.csv (0-1 scale)
temp = float(temp_c) / 41.0
hum = float(humidity) / 100.0
windspeed = float(windspeed_kmh) / 67.0

# atemp not provided by user; approximate with temp (fine for deployment)
atemp = temp

# ---------------- Feature engineering (MUST match training) ----------------
is_peak_hour = 1 if hr in [7, 8, 9, 17, 18, 19] else 0
is_weekend = 1 if weekday in [0, 6] else 0
temp_hum_interaction = temp * hum

# ---------------- Main UI ----------------
st.markdown('<div class="main-title">🚴 BikeShare Demand Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Hourly Bike Rental Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="author"><b>Hassan Khan Alizai (225187)</b> – Air University</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict = st.button("🧠 Predict Hourly Demand")

    if predict:
        # Build raw input dict (NO dteday here, because training dropped it)
        input_data = {
            "season": season,
            "yr": yr,
            "mnth": mnth,
            "hr": hr,
            "holiday": holiday,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weathersit,
            "temp": temp,
            "atemp": atemp,
            "hum": hum,
            "windspeed": windspeed,
            "is_peak_hour": is_peak_hour,
            "is_weekend": is_weekend,
            "temp_hum_interaction": temp_hum_interaction
        }

        # Create input df
        input_df = pd.DataFrame([input_data])

        # Ensure EXACT columns order and ensure all expected columns exist
        # If anything missing, create it as NaN (imputer in pipeline will handle)
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = np.nan

        input_df = input_df[model_columns]

        # Predict (model outputs log1p(cnt))
        pred_log = model.predict(input_df)[0]
        prediction = int(round(np.expm1(pred_log)))
        prediction = max(prediction, 0)  # safety clamp

        st.markdown(
            f'<div class="result-box">Predicted Demand: {prediction} bikes/hour</div>',
            unsafe_allow_html=True
        )

        if prediction < 100:
            st.markdown('<div class="status-box">✅ Normal demand — System running smoothly</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box">⚠ High demand — Consider adding bikes</div>', unsafe_allow_html=True)

        st.markdown('<div class="small-note">Note: Prediction is an estimate based on historical patterns.</div>', unsafe_allow_html=True)

        feedback_url = "https://forms.gle/cJa7pw2hMXyb4ac78"
        st.markdown(f"""
        <div style="text-align:center; margin-top:30px;">
            <a href="{feedback_url}" target="_blank">
                <button style="
                    background-color:#2b7de9;
                    color:white;
                    font-size:20px;
                    padding:12px 24px;
                    border:none;
                    border-radius:10px;
                    cursor:pointer;
                ">
                    📝 Give Feedback
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

        if debug:
            st.subheader("Debug")
            st.write("Model expects columns:", model_columns)
            st.write("Input dataframe sent to model:")
            st.dataframe(input_df)
