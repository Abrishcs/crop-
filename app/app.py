import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime

from cleaning import CANONICAL_CROPS, CANONICAL_REGIONS, MONTH_NUM, MONTH_ORDER
from features import MODEL_FEATURES, season_features_for_one_plot

APP_DIR = os.path.dirname(__file__)
ASSETS = os.path.join(APP_DIR, "assets")
MODELS = os.path.join(APP_DIR, "..", "models")
PROCESSED = os.path.join(APP_DIR, "..", "data", "processed")

st.set_page_config(page_title="Ethiopian Crop Yield Predictor", layout="wide", initial_sidebar_state="collapsed")

# Initialize Session State for History & Presets
if "history" not in st.session_state:
    st.session_state["history"] = []

if "preset" not in st.session_state:
    st.session_state["preset"] = None

# Custom CSS matching the high-end Ethiopian dark-green UI design with agricultural landscape backdrop
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Ethiopic:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans Ethiopic', sans-serif;
    }
    
    /* Transparent Header */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Background with lush agricultural landscape backdrop */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-image: linear-gradient(180deg, rgba(7, 20, 12, 0.65) 0%, rgba(4, 12, 7, 0.88) 100%), 
                          url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?q=80&w=1600&auto=format&fit=crop') !important;
        background-size: cover !important;
        background-position: center top !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        color: #e2e8f0 !important;
    }

    /* Remove extra padding top */
    .block-container {
        padding-top: 1.2rem !important;
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
        margin-bottom: 16px;
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

    /* Streamlit Tabs Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
        padding-bottom: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0px 22px;
        backdrop-filter: blur(10px);
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        border-color: #22c55e;
        color: #4ade80;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(34, 197, 94, 0.25) !important;
        border-color: #22c55e !important;
        color: #4ade80 !important;
        box-shadow: 0 0 15px rgba(34, 197, 94, 0.25);
    }

    /* Form Container (Left Card) */
    div[data-testid="stForm"] {
        background: rgba(13, 26, 17, 0.82) !important;
        border: 1px solid rgba(74, 222, 128, 0.25) !important;
        backdrop-filter: blur(20px);
        border-radius: 20px !important;
        padding: 24px 28px !important;
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
        margin-bottom: 16px;
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
        margin-top: 10px !important;
        letter-spacing: 0.3px;
    }

    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        box-shadow: 0 6px 28px rgba(245, 158, 11, 0.7) !important;
        transform: translateY(-2px);
    }

    /* Preset Buttons */
    .preset-btn-container {
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
    }

    /* Right Result Card Glass Box */
    .result-container {
        background: rgba(13, 26, 17, 0.82);
        border: 1px solid rgba(74, 222, 128, 0.25);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 24px 28px;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.6);
        height: 100%;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 20px;
    }

    .metric-card-green {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 16px;
        padding: 16px 18px;
    }

    .metric-card-gold {
        background: rgba(234, 179, 8, 0.12);
        border: 1px solid rgba(234, 179, 8, 0.4);
        border-radius: 16px;
        padding: 16px 18px;
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

# Top Header Bar
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

# Interactive Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard / መቆጣጠሪያ", 
    "📜 History / ታሪክ", 
    "📈 Regional Insights / ክልላዊ መረጃ", 
    "⚙️ What-If Simulator / ማስመሰያ"
])

with tab1:
    # Quick Interactive Preset Buttons
    st.markdown("##### ⚡ Quick Interactive Presets:")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("🌾 Oromia Teff Plot"):
            st.session_state["preset"] = "oromia_teff"
    with p_col2:
        if st.button("🌽 Amhara Maize Plot"):
            st.session_state["preset"] = "amhara_maize"
    with p_col3:
        if st.button("🌾 SNNPR Wheat Plot"):
            st.session_state["preset"] = "snnpr_wheat"
    with p_col4:
        if st.button("🔄 Reset Defaults"):
            st.session_state["preset"] = None

    # Handle preset values
    def_region, def_crop, def_rain, def_fert, def_seed = "amhara", "teff", 800.0, 100.0, False
    if st.session_state.get("preset") == "oromia_teff":
        def_region, def_crop, def_rain, def_fert, def_seed = "oromia", "teff", 950.0, 120.0, True
    elif st.session_state.get("preset") == "amhara_maize":
        def_region, def_crop, def_rain, def_fert, def_seed = "amhara", "maize", 1100.0, 150.0, True
    elif st.session_state.get("preset") == "snnpr_wheat":
        def_region, def_crop, def_rain, def_fert, def_seed = "snnpr", "wheat", 850.0, 90.0, False

    # Two-column Dashboard Layout
    left_col, right_col = st.columns([1, 1.05], gap="large")

    with left_col:
        with st.form("plot_form"):
            st.markdown('<div class="card-header-green">Input Crop Data</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-header-amharic">የሰብል መረጃ ያስገቡ</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                region_idx = CANONICAL_REGIONS.index(def_region) if def_region in CANONICAL_REGIONS else 0
                crop_idx = CANONICAL_CROPS.index(def_crop) if def_crop in CANONICAL_CROPS else 0
                region = st.selectbox("Region / ወረዳ", CANONICAL_REGIONS, index=region_idx, format_func=str.capitalize)
                crop_type = st.selectbox("Crop Type / የሰብል ዓይነት", CANONICAL_CROPS, index=crop_idx, format_func=str.capitalize)
                survey_year = st.number_input("Survey Year / ዓመት", min_value=2021, max_value=2026, value=2024, step=1)
                planting_month = st.selectbox("Planting Month / የመትከያ ወር", MONTH_ORDER, format_func=str.capitalize)
                altitude_m = st.number_input("Altitude (m) / ከፍታ", min_value=0.0, max_value=4500.0, value=1800.0, step=50.0)
                farm_size_ha = st.number_input("Land Size (ha) / የመሬት መጠን", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
            with c2:
                rainfall_mm_season = st.number_input("Season Rainfall (mm)", min_value=0.0, max_value=5000.0, value=def_rain, step=10.0)
                fertilizer_kg_per_ha = st.number_input("Fertilizer (kg/ha) / ማዳበሪያ", min_value=0.0, max_value=1000.0, value=def_fert, step=5.0)
                soil_quality_index = st.slider("Soil Quality / የአፈር ጥራት", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                labor_days_per_ha = st.number_input("Labor (days/ha) / ጉልበት", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
                distance_to_market_km = st.number_input("Distance to Market (km)", min_value=0.0, max_value=300.0, value=15.0, step=1.0)
                
                f1, f2 = st.columns(2)
                with f1:
                    improved_seed_used = st.checkbox("Improved seed", value=def_seed)
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

                # Save to Interactive Session History
                st.session_state["history"].append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Region": region.capitalize(),
                    "Crop": crop_type.capitalize(),
                    "Land Size (ha)": farm_size_ha,
                    "Predicted Yield (T/ha)": round(predicted_yield, 2),
                    "Est. Revenue (Birr)": round(revenue, 0),
                })

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

# TAB 2: HISTORY
with tab2:
    st.markdown("### 📜 Prediction History Log / የትንበያ ታሪክ")
    if len(st.session_state["history"]) > 0:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True)
        
        h_col1, h_col2 = st.columns([1, 4])
        with h_col1:
            csv_data = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download History CSV", data=csv_data, file_name="prediction_history.csv", mime="text/csv")
        with h_col2:
            if st.button("🗑️ Clear History"):
                st.session_state["history"] = []
                st.rerun()
    else:
        st.info("No predictions recorded yet in this session. Run predictions in the Dashboard tab to log them here!")

# TAB 3: REGIONAL INSIGHTS
with tab3:
    st.markdown("### 📈 Ethiopian Agricultural Regional Insights / ክልላዊ መረጃዎች")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        sel_insight_region = st.selectbox("Select Region for Analysis:", CANONICAL_REGIONS, format_func=str.capitalize, key="insight_reg")
        region_df = master_train[master_train["region"] == sel_insight_region]
        
        if not region_df.empty:
            avg_yield_by_crop = region_df.groupby("crop_type")["yield_tons_per_ha"].mean()
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            avg_yield_by_crop.plot(kind='bar', ax=ax, color='#22c55e', edgecolor='#86efac', linewidth=1.2)
            ax.set_title(f"Average Yield by Crop in {sel_insight_region.capitalize()} (Tons/Ha)", color='white', fontsize=11, pad=10)
            ax.set_ylabel("Tons / Ha", color='#cbd5e1')
            ax.tick_params(colors='#94a3b8')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#ffffff20')
            ax.spines['bottom'].set_color('#ffffff20')
            ax.grid(axis='y', linestyle='--', alpha=0.15, color='#ffffff')
            st.pyplot(fig)
    
    with r_col2:
        st.markdown("#### 💡 Regional Summary Highlights")
        st.markdown(f"""
        - **Total Surveys Analyzed:** `{len(region_df):,}` smallholder plots in {sel_insight_region.capitalize()}.
        - **Highest Yielding Crop:** `{avg_yield_by_crop.idxmax().capitalize() if not region_df.empty else 'N/A'}` ({avg_yield_by_crop.max():.2f} T/Ha).
        - **Average Regional Rainfall:** `{region_df['rainfall_mm_season'].mean():.0f} mm` per growing season.
        - **Improved Seed Adoption Rate:** `{region_df['improved_seed_used'].mean()*100:.1f}%` of plots.
        """)

# TAB 4: WHAT-IF SIMULATOR
with tab4:
    st.markdown("### ⚙️ Interactive What-If Optimization Simulator / ማስመሰያ")
    st.caption("Simulate how adding fertilizer or upgrading seeds boosts your expected yield and net revenue in real-time!")
    
    sim_col1, sim_col2 = st.columns([1, 1])
    
    with sim_col1:
        st.markdown("##### Baseline Inputs")
        sim_region = st.selectbox("Simulated Region", CANONICAL_REGIONS, format_func=str.capitalize, key="sim_reg")
        sim_crop = st.selectbox("Simulated Crop", CANONICAL_CROPS, format_func=str.capitalize, key="sim_crop")
        sim_fert_base = st.slider("Current Fertilizer (kg/ha)", 0, 300, 50, step=10)
        sim_seed_base = st.checkbox("Currently Using Improved Seed?", value=False)
        
        st.markdown("##### 🚀 Intervention Boost")
        extra_fert = st.slider("Add Extra Fertilizer (+ kg/ha)", 0, 200, 50, step=10)
        upgrade_seed = st.checkbox("Upgrade to Certified Improved Seed", value=True)
        
    with sim_col2:
        try:
            # Baseline Prediction
            base_row = {
                "region": sim_region, "crop_type": sim_crop, "planting_month": "may",
                "altitude_m": 1800.0, "rainfall_mm_season": 900.0, "farm_size_ha": 1.0,
                "fertilizer_kg_per_ha": sim_fert_base, "improved_seed_used": int(sim_seed_base),
                "pest_disease_flag": 0, "soil_quality_index": 0.5, "labor_days_per_ha": 40.0,
                "distance_to_market_km": 15.0, "season_avg_temp_c": 21.0,
                "season_rainfall_mm_total": 900.0, "season_extreme_heat_days": 2.0,
                "season_months_matched": 4, "temp_deviation_from_region_avg": 0.0,
                "fertilizer_x_improved_seed": sim_fert_base * int(sim_seed_base),
                "planting_month_num": 5, "survey_year": 2024
            }
            base_yield = max(float(model.predict(pd.DataFrame([base_row])[MODEL_FEATURES])[0]), 0.0)
            
            # Boosted Prediction
            new_fert = sim_fert_base + extra_fert
            new_seed = int(sim_seed_base or upgrade_seed)
            boost_row = base_row.copy()
            boost_row["fertilizer_kg_per_ha"] = new_fert
            boost_row["improved_seed_used"] = new_seed
            boost_row["fertilizer_x_improved_seed"] = new_fert * new_seed
            
            boosted_yield = max(float(model.predict(pd.DataFrame([boost_row])[MODEL_FEATURES])[0]), 0.0)
            yield_lift = boosted_yield - base_yield
            pct_lift = (yield_lift / base_yield * 100) if base_yield > 0 else 0
            
            st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 16px; padding: 20px; margin-top: 20px;">
                <div style="font-size: 0.9rem; color: #94a3b8;">Simulated Yield Lift</div>
                <div style="font-size: 2.8rem; font-weight: 800; color: #4ade80;">+{yield_lift:.2f} T/Ha</div>
                <div style="font-size: 1rem; color: #86efac; font-weight: 600;">(+{pct_lift:.1f}% Increase over Baseline)</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Matplotlib Visual Comparison
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            bars = ax.bar(["Baseline", "Boosted"], [base_yield, boosted_yield], color=["#94a3b8", "#22c55e"], width=0.4)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f} T/Ha', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='white', fontweight='bold')
            ax.set_ylabel("Yield (T/Ha)", color='#cbd5e1')
            ax.tick_params(colors='#94a3b8')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#ffffff20')
            ax.spines['bottom'].set_color('#ffffff20')
            st.pyplot(fig)
            
        except Exception as exc:
            st.error(f"Simulator error: {exc}")
