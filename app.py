import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TRIADA-IR Calculator",
    page_icon="🩸",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        font-weight: bold;
    }
    .sub-text {
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================
MODEL_FILE = 'triada_ir_model.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

model = load_model()

# ==========================================
# 3. UI LAYOUT
# ==========================================
st.markdown('<div class="main-header">🩸 TRIADA-IR Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Machine Learning Estimation of Insulin Resistance Risk</div>', unsafe_allow_html=True)

if model is None:
    st.error(f"⚠️ Model file '{MODEL_FILE}' not found! Please run the training script first to generate it.")
    st.stop()

# --- INPUT FORM ---
with st.container():
    st.write("### 📝 Patient Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=45, step=1)
        gender_label = st.selectbox("Gender", ["Male", "Female"])
        # Convert Gender to Code (1=Male, 2=Female)
        gender = 1 if gender_label == "Male" else 2

    with col2:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)

    glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50.0, max_value=400.0, value=95.0, step=1.0)

    # Calculate BMI
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    st.info(f"📊 Calculated BMI: **{bmi:.1f} kg/m²**")

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button("🚀 Calculate Risk", use_container_width=True):
    
    # Prepare input DataFrame (Must match training columns exactly)
    input_data = pd.DataFrame({
        "RIDAGEYR": [age],
        "RIAGENDR": [gender],
        "BMXBMI":   [bmi],
        "BMXWT":    [weight],
        "LBXGLU":   [glucose]
    })

    # Predict Probability
    score = model.predict_proba(input_data)[:, 1][0]
    
    # Define Categories based on Percentiles
    if score < 0.20:
        risk_level = "LOW RISK"
        color = "#2ECC71" # Green
        msg = "Patient is likely Insulin Sensitive."
    elif score < 0.50:
        risk_level = "NORMAL / BORDERLINE"
        color = "#F1C40F" # Yellow/Orange
        msg = "Moderate risk. Lifestyle monitoring recommended."
    else:
        risk_level = "HIGH RISK"
        color = "#E74C3C" # Red
        msg = "High likelihood of Insulin Resistance."

    # Display Result
    st.markdown(f"""
        <div class="result-box" style="background-color: {color}20; border: 2px solid {color};">
            <h2 style="color: {color}; margin:0;">{risk_level}</h2>
            <h1 style="color: #333; font-size: 3.5rem;">{score:.3f}</h1>
            <p style="font-size: 1.2rem;">{msg}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show interpretation guide
    with st.expander("ℹ️  How to interpret this score?"):
        st.write("""
        - **< 0.200**: Low probability of Insulin Resistance.
        - **0.200 - 0.500**: Average population risk.
        - **> 0.500**: High probability (Top 20% of risk distribution).
        """)

# Footer
st.markdown("---")
st.caption("Developed for Research Use Only. Not a medical diagnosis tool.")