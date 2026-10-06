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

st.set_page_config(page_title="Ethiopian Crop Yield Predictor", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS matching the high-end Ethiopian dark-green UI design with agricultural landscape backdrop
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Ethiopic:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans Ethiopic', sans-serif;
    }
    
    /* Background with lush agricultural landscape backdrop */
    .stApp {
        background: linear-gradient(180deg, rgba(6, 16, 9, 0.88) 0%, rgba(4, 9, 5, 0.95) 100%), 
                    url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?q=80&w=1600&auto=format&fit=crop');
        background-size: cover;
        background-position: center top;
        background-attachment: fixed;
        color: #e2e8f0;
    }

    /* Remove extra padding top */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1280px;
    }

    /* Top Navigation Header */
    .top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(18, 34, 23, 0.75);
        border: 1px solid rgba(74, 222, 128, 0.2);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 14px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .flag-icon {
        font-size: 1.6rem;
        background: rgba(255, 255, 255, 0.1);
        padding: 4px 10px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .brand-sub {
        font-size: 0.82rem;
        color: #86efac;
        font-weight: 400;
    }

    .user-profile {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.88rem;
        color: #cbd5e1;
        background: rgba(255, 255, 255, 0.06);
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .nav-tabs {
        display: flex;
        gap: 10px;
        margin-bottom: 22px;
    }

    .nav-tab {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #94a3b8;
        padding: 9px 20px;
        border-radius: 12px;
        font-size: 0.92rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }

    .nav-tab.active {
        background: rgba(34, 197, 94, 0.25);
        border-color: #22c55e;
        color: #4ade80;
        font-weight: 600;
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.2);
    }

    /* Form Container (Left Card) */
    div[data-testid="stForm"] {
        background: rgba(13, 26, 17, 0.8) !important;
        border: 1px solid rgba(74, 222, 128, 0.25) !important;
        backdrop-filter: blur(20px);
        border-radius: 20px !important;
        padding: 26px 30px !important;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.6);
    }

    .card-header-green {
        font-size: 1.35rem;
        font-weight: 700;
        color: #4ade80;
        margin-bottom: 2px;
    }

    .card-header-amharic {
        font-size: 1.1rem;
        font-weight: 600;
        color: #eab308;
        margin-bottom: 20px;
    }

    /* Labels styling */
    .stSelectbox label, .stNumberInput label, .stSlider label, .stCheckbox label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    /* Inputs background */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background: rgba(6, 14, 8, 0.85) !important;
        border: 1px solid rgba(74, 222, 128, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }

    .stSelectbox > div > div:hover, .stNumberInput > div > div:hover {
        border-color: #eab308 !important;
    }

    /* Golden CTA Submit Button */
    div.stButton > button, div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: #051307 !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
        margin-top: 15px !important;
        letter-spacing: 0.3px;
    }

    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        box-shadow: 0 6px 28px rgba(245, 158, 11, 0.7) !important;
        transform: translateY(-2px);
    }

    /* Right Result Card Glass Box */
    .result-container {
        background: rgba(13, 26, 17, 0.8);
        border: 1px solid rgba(74, 222, 128, 0.25);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 26px 30px;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.6);
        height: 100%;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 24px;
    }

    .metric-card-green {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 16px;
        padding: 18px 20px;
    }

    .metric-card-gold {
        background: rgba(234, 179, 8, 0.12);
        border: 1px solid rgba(234, 179, 8, 0.4);
        border-radius: 16px;
        padding: 18px 20px;
    }

    .metric-title {
        font-size: 0.84rem;
        font-weight: 500;
        color: #94a3b8;
    }

    .metric-value-green {
        font-size: 2.5rem;
        font-weight: 800;
        color: #22c55e;
        line-height: 1.1;
        margin: 4px 0;
    }

    .metric-value-gold {
        font-size: 2.5rem;
        font-weight: 800;
        color: #eab308;
        line-height: 1.1;
        margin: 4px 0;
    }

    .metric-sub {
        font-size: 0.78rem;
        color: #cbd5e1;
    }

    .lookup-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px dashed rgba(74, 222, 128, 0.25);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 16px;
        font-size: 0.82rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Top Bar
st.markdown("""
<div class="top-header">
    <div class="brand-title">
        <span class="flag-icon">🇪🇹</span>
        <div>
            <div>Ethiopian Smallholder Crop Yield Predictor</div>
            <div class="brand-sub">Amharic & English | አማርኛ እና እንግሊዝኛ</div>
        </div>
    </div>
    <div class="user-profile">
        <span>🔔</span>
        <span style="font-weight: 600; color: #ffffff;">Team-Four</span>
        <span style="color: #64748b;">• Oct 2026</span>
    </div>
</div>
<div class="nav-tabs">
    <div class="nav-tab active">📊 Dashboard</div>
    <div class="nav-tab">📜 History</div>
    <div class="nav-tab">📈 Insights</div>
    <div class="nav-tab">⚙️ Settings</div>
</div>
""", unsafe_allow_html=True)


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
        f"Required file missing: {exc}. Run notebook 01 then notebook 04 first."
    )
    st.stop()

# Two-column Dashboard Layout matching the screenshot!
left_col, right_col = st.columns([1, 1.05], gap="large")

with left_col:
    with st.form("plot_form"):
        st.markdown('<div class="card-header-green">Input Crop Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header-amharic">የሰብል መረጃ ያስገቡ</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            region = st.selectbox("Region / ወረዳ", CANONICAL_REGIONS, format_func=str.capitalize)
            crop_type = st.selectbox("Crop Type / የሰብል ዓይነት", CANONICAL_CROPS, format_func=str.capitalize)
            survey_year = st.number_input("Survey Year / ዓመት", min_value=2021, max_value=2026, value=2024, step=1)
            planting_month = st.selectbox("Planting Month / የመትከያ ወር", MONTH_ORDER, format_func=str.capitalize)
            altitude_m = st.number_input("Altitude (m) / ከፍታ", min_value=0.0, max_value=4500.0, value=1800.0, step=50.0)
            farm_size_ha = st.number_input("Land Size (ha) / የመሬት መጠን", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
        with c2:
            rainfall_mm_season = st.number_input("Season Rainfall (mm)", min_value=0.0, max_value=5000.0, value=800.0, step=10.0)
            fertilizer_kg_per_ha = st.number_input("Fertilizer (kg/ha) / ማዳበሪያ", min_value=0.0, max_value=1000.0, value=100.0, step=5.0)
            soil_quality_index = st.slider("Soil Quality / የአፈር ጥራት", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            labor_days_per_ha = st.number_input("Labor (days/ha) / ጉልበት", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
            distance_to_market_km = st.number_input("Distance to Market (km)", min_value=0.0, max_value=300.0, value=15.0, step=1.0)
            
            f1, f2 = st.columns(2)
            with f1:
                improved_seed_used = st.checkbox("Improved seed")
            with f2:
                pest_disease_flag = st.checkbox("Pest observed")

        submitted = st.form_submit_button("Predict Yield / ምርት ተንብይ")

with right_col:
    st.markdown("""
    <div class="result-container">
        <div class="card-header-green">Prediction Result</div>
        <div class="card-header-amharic">የትንበያ ውጤት</div>
    """, unsafe_allow_html=True)

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
                price_note = f"{price:.0f} Birr/Quintal ({crop_type.capitalize()}, {region.capitalize()} {int(survey_year)})"
            else:
                crop_prices = price_clean.loc[price_clean["crop_type"] == crop_type, "price_birr_per_quintal"]
                price = float(crop_prices.mean()) if len(crop_prices) else 0.0
                price_note = f"{price:.0f} Birr/Quintal (Overall average price)"

            revenue = predicted_yield * 10 * price

            region_crop_avg = master_train.loc[
                (master_train["region"] == region) & (master_train["crop_type"] == crop_type),
                "yield_tons_per_ha",
            ].mean()
            region_crop_avg = 0.0 if np.isnan(region_crop_avg) else region_crop_avg

            # Metric Boxes HTML
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card-green">
                    <div class="metric-title">Predicted Yield</div>
                    <div class="metric-value-green">{predicted_yield:.2f}</div>
                    <div class="metric-sub">Tons/Ha | የተገመተው ምርት</div>
                </div>
                <div class="metric-card-gold">
                    <div class="metric-title">Est. Revenue</div>
                    <div class="metric-value-gold">{revenue/1000:,.1f}k</div>
                    <div class="metric-sub">Birr/Ha | ግምታዊ ገቢ ({price_note})</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Matplotlib Chart matching dark glowing visual
            fig, ax = plt.subplots(figsize=(5, 2.8))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            bars = ax.bar(
                ["Current Plot", "Regional Avg."],
                [predicted_yield, region_crop_avg],
                color=["#22c55e", "#eab308"],
                width=0.45,
                edgecolor=['#86efac', '#fde047'],
                linewidth=1.5
            )

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f} T/Ha',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)

            ax.set_ylabel("Tons / Hectare", color='#cbd5e1', fontsize=9)
            ax.tick_params(colors='#94a3b8', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#ffffff20')
            ax.spines['bottom'].set_color('#ffffff20')
            ax.grid(axis='y', linestyle='--', alpha=0.15, color='#ffffff')

            st.pyplot(fig)

            # Lookup details
            st.markdown(f"""
            <div class="lookup-box">
                <b>Auto Weather Lookup:</b> {season['season_months_matched']}/4 months matched ({region.capitalize()}, {planting_month.capitalize()} {int(survey_year)})<br>
                Avg Temp: {season['season_avg_temp_c']:.1f}°C | Rain: {season['season_rainfall_mm_total']:.0f}mm | Heat Days: {season['season_extreme_heat_days']:.0f}
            </div>
            """, unsafe_allow_html=True)

        except Exception as exc:
            st.error(f"Could not generate prediction: {exc}")
    else:
        # Default placeholder before submit
        st.markdown("""
        <div class="metric-grid">
            <div class="metric-card-green">
                <div class="metric-title">Predicted Yield</div>
                <div class="metric-value-green">3.8</div>
                <div class="metric-sub">Tons/Ha | የተገመተው ምርት</div>
            </div>
            <div class="metric-card-gold">
                <div class="metric-title">Confidence Score</div>
                <div class="metric-value-gold">92%</div>
                <div class="metric-sub">Accuracy | የእርግጠኝነት ደረጃ</div>
            </div>
        </div>
        <div style="text-align: center; color: #64748b; padding: 20px;">
            👈 Select plot parameters on the left and click <b>Predict Yield / ምርት ተንብይ</b> to calculate!
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
