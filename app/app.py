import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from cleaning import CANONICAL_CROPS, CANONICAL_REGIONS, MONTH_NUM, MONTH_ORDER
from features import MODEL_FEATURES, season_features_for_one_plot

APP_DIR = os.path.dirname(__file__)
ASSETS = os.path.join(APP_DIR, "assets")
MODELS = os.path.join(APP_DIR, "..", "models")
PROCESSED = os.path.join(APP_DIR, "..", "data", "processed")
st.set_page_config(page_title="Ethiopian Crop Yield Predictor", layout="centered")

st.markdown("""
<style>
    /* Premium Dark Mode & Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Earthy Deep Green Gradient Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(14, 30, 20) 0%, rgb(10, 15, 12) 90%);
        color: #e2e8f0;
    }
    
    /* Headers & Text */
    h1 {
        background: -webkit-linear-gradient(45deg, #FFD700, #4ade80);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 0px !important;
    }
    
    /* Glassmorphism Input Cards */
    div[data-testid="stForm"] {
        background: rgba(30, 41, 35, 0.4);
        border: 1px solid rgba(74, 222, 128, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Soft Rounded Input Fields */
    .stSelectbox > div > div, .stNumberInput > div > div, .stTextInput > div > div {
        background-color: rgba(20, 30, 25, 0.6) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 12px;
        color: white !important;
    }

    /* Glowing Predict Button */
    button[data-testid="baseButton-formSubmit"] {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 10px 24px !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.5) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: 15px;
    }
    button[data-testid="baseButton-formSubmit"]:hover {
        box-shadow: 0 0 25px rgba(74, 222, 128, 0.8) !important;
        transform: scale(1.02);
    }
    
    /* Result Success Messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border-left: 5px solid #10b981 !important;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    .stInfo {
        background: rgba(255, 215, 0, 0.1) !important;
        border-left: 5px solid #FFD700 !important;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
</style>
""", unsafe_allow_html=True)

st.title("Ethiopian Smallholder Crop Yield Predictor")
st.caption(
    "Enter your plot details. Weather and price are looked up automatically from the "
    "region, crop, and year you pick -- you never type those in."
)


@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(MODELS, "final_model.joblib"))
    weather_clean = pd.read_csv(os.path.join(ASSETS, "weather_clean.csv"))
    price_clean = pd.read_csv(os.path.join(ASSETS, "price_clean.csv"))
    master_train = pd.read_csv(os.path.join(PROCESSED, "master_train.csv"))
    return model, weather_clean, price_clean, master_train


try:
    model, weather_clean, price_clean, master_train = load_assets()
except FileNotFoundError as exc:
    st.error(
        f"Required file missing: {exc}. Run notebook 01 then notebook 04 first, so "
        "models/final_model.joblib and app/assets/weather_clean.csv + price_clean.csv exist."
    )
    st.stop()

with st.form("plot_form"):
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Region", CANONICAL_REGIONS, format_func=str.capitalize)
        crop_type = st.selectbox("Crop", CANONICAL_CROPS, format_func=str.capitalize)
        survey_year = st.number_input("Year", min_value=2021, max_value=2026, value=2024, step=1)
        planting_month = st.selectbox("Planting month", MONTH_ORDER, format_func=str.capitalize)
        altitude_m = st.number_input("Altitude (m)", min_value=0.0, max_value=4500.0, value=1800.0, step=50.0)
        farm_size_ha = st.number_input("Farm size (ha)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
        rainfall_mm_season = st.number_input(
            "Your estimated season rainfall (mm)", min_value=0.0, max_value=5000.0, value=800.0, step=10.0
        )
    with col2:
        fertilizer_kg_per_ha = st.number_input("Fertilizer (kg/ha)", min_value=0.0, max_value=1000.0, value=100.0, step=5.0)
        improved_seed_used = st.checkbox("Improved seed used")
        pest_disease_flag = st.checkbox("Pest/disease observed")
        soil_quality_index = st.slider("Soil quality index", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        labor_days_per_ha = st.number_input("Labor (days/ha)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
        distance_to_market_km = st.number_input("Distance to market (km)", min_value=0.0, max_value=300.0, value=15.0, step=1.0)

    submitted = st.form_submit_button("Predict yield")

if submitted:
    try:
        season = season_features_for_one_plot(weather_clean, region, int(survey_year), planting_month)

        row = {
            "region": region,
            "crop_type": crop_type,
            "planting_month": planting_month,
            "altitude_m": altitude_m,
            "rainfall_mm_season": rainfall_mm_season,
            "farm_size_ha": farm_size_ha,
            "fertilizer_kg_per_ha": fertilizer_kg_per_ha,
            "improved_seed_used": int(improved_seed_used),
            "pest_disease_flag": int(pest_disease_flag),
            "soil_quality_index": soil_quality_index,
            "labor_days_per_ha": labor_days_per_ha,
            "distance_to_market_km": distance_to_market_km,
            "season_avg_temp_c": season["season_avg_temp_c"],
            "season_rainfall_mm_total": season["season_rainfall_mm_total"],
            "season_extreme_heat_days": season["season_extreme_heat_days"],
            "season_months_matched": season["season_months_matched"],
            "temp_deviation_from_region_avg": season["temp_deviation_from_region_avg"],
            "fertilizer_x_improved_seed": fertilizer_kg_per_ha * int(improved_seed_used),
            "planting_month_num": MONTH_NUM[planting_month],
            "survey_year": int(survey_year),
        }
        X_input = pd.DataFrame([row])[MODEL_FEATURES]

        predicted_yield = max(float(model.predict(X_input)[0]), 0.0)

        price_match = price_clean[
            (price_clean["region"] == region)
            & (price_clean["crop_type"] == crop_type)
            & (price_clean["year"] == int(survey_year))
        ]
        if not price_match.empty:
            price = float(price_match["price_birr_per_quintal"].iloc[0])
            price_note = (
                f"{price:.0f} birr/quintal (exact match for {crop_type.capitalize()} "
                f"in {region.capitalize()}, {int(survey_year)})"
            )
        else:
            crop_prices = price_clean.loc[price_clean["crop_type"] == crop_type, "price_birr_per_quintal"]
            price = float(crop_prices.mean()) if len(crop_prices) else 0.0
            price_note = (
                f"{price:.0f} birr/quintal (no exact region/year match; using "
                f"{crop_type.capitalize()}'s overall average price)"
            )

        revenue = predicted_yield * 10 * price

        st.success(f"Predicted yield: {predicted_yield:.2f} tons/ha")
        st.info(f"Estimated revenue: {revenue:,.0f} birr/ha")

        st.markdown("**What was looked up for you:**")
        if season["season_months_matched"] > 0:
            st.write(
                f"- Season weather: {season['season_months_matched']}/4 months matched in "
                f"{region.capitalize()} starting {planting_month.capitalize()} {int(survey_year)} -> "
                f"avg temp {season['season_avg_temp_c']:.1f} C, "
                f"total rainfall {season['season_rainfall_mm_total']:.0f} mm, "
                f"{season['season_extreme_heat_days']:.0f} extreme-heat days."
            )
        else:
            st.write(
                f"- No weather records matched that exact region/year/month window; "
                f"fell back to {region.capitalize()}'s overall climate average."
            )
        st.write(f"- Price: {price_note}")

        region_crop_avg = master_train.loc[
            (master_train["region"] == region) & (master_train["crop_type"] == crop_type),
            "yield_tons_per_ha",
        ].mean()
        region_crop_avg = 0.0 if np.isnan(region_crop_avg) else region_crop_avg

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(
            ["Your plot", f"{region.capitalize()} {crop_type.capitalize()} avg"],
            [predicted_yield, region_crop_avg],
            color=["#4C72B0", "#937860"],
        )
        ax.set_ylabel("Yield (tons/ha)")
        ax.set_title("Your prediction vs. historical regional average")
        st.pyplot(fig)

    except Exception as exc:
        st.error(f"Could not generate a prediction: {exc}")
