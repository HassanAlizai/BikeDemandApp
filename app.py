import streamlit as st
import pandas as pd
import joblib

# ---------------- Load model ----------------
model = joblib.load("random_forest_bike_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="BikeShare Demand Predictor", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar Inputs ----------------
st.sidebar.title("Enter Conditions")

hr = st.sidebar.slider("Hour", 0, 23, 8)
windspeed_kmh = st.sidebar.slider("Wind (km/h)", 0, 67, 10)
temp_c = st.sidebar.slider("Temp (°C)", 0, 40, 28)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 37)

day_name = st.sidebar.selectbox("Day", ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
season_name = st.sidebar.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
weather_name = st.sidebar.selectbox("Weather", ["Clear", "Mist", "Light Rain", "Heavy Rain"])

# ---------------- Mapping ----------------
weekday_map = {"Sun":0,"Mon":1,"Tue":2,"Wed":3,"Thu":4,"Fri":5,"Sat":6}
season_map = {"Spring":1,"Summer":2,"Fall":3,"Winter":4}
weather_map = {"Clear":1,"Mist":2,"Light Rain":3,"Heavy Rain":4}

weekday = weekday_map[day_name]
season = season_map[season_name]
weathersit = weather_map[weather_name]

# Normalization (match training)
temp = temp_c / 41
hum = humidity / 100
windspeed = windspeed_kmh / 67
atemp = temp
yr = 1
mnth = 6
holiday = 0
workingday = 1 if weekday in [1,2,3,4,5] else 0

# Feature engineering
month = mnth
is_peak_hour = 1 if hr in [7,8,9,17,18,19] else 0
is_weekday = 1 if weekday in [1,2,3,4,5] else 0
temp_squared = temp ** 2
temp_hum_interaction = temp * hum

# ---------------- Main UI ----------------
st.markdown('<div class="main-title">🚴 BikeShare Demand Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Hourly Bike Rental Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="author"><b>Hassan Khan Alizai (225187)</b> – Air University</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict = st.button("🧠 Predict Hourly Demand")

    if predict:
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
            "month": month,
            "is_peak_hour": is_peak_hour,
            "is_weekday": is_weekday,
            "temp_squared": temp_squared,
            "temp_hum_interaction": temp_hum_interaction
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df[model_columns]

        prediction = int(model.predict(input_df)[0])

        st.markdown(
            f'<div class="result-box">Predicted Demand: {prediction} bikes/hour</div>',
            unsafe_allow_html=True
        )

        if prediction < 100:
            st.markdown('<div class="status-box">✅ Normal demand — System running smoothly</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box">⚠ High demand — Consider adding bikes</div>', unsafe_allow_html=True)

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
