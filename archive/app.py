"""
DiabetesCheck AI - Clinical Decision Support System (DSS)
Author: Arthur PREVEL
Version: 1.1.3 - Final Render & Contrast Fix
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import base64

# =============================================
# 1. PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="DiabetesCheck AI - Clinical DSS",
    page_icon="images/favicon.ico",
    layout="wide"
)

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

logo_base64 = get_base64_image("images/logo.png")

# =============================================
# 2. CUSTOM CSS (Force Visibility & Styles)
# =============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1e293b; }

[data-testid="stSidebar"] { background-color: #f1f5f9 !important; }
[data-testid="stSidebar"] * { color: #1e293b !important; }

/* Metrics Cards */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    padding: 15px !important;
    border-radius: 10px !important;
}

/* Button UI */
.stButton>button {
    width: 100%; border-radius: 8px; background-color: #1e1b4b;
    color: white !important; font-weight: 600; height: 3.5rem; border: none;
}
.stButton>button:hover { background-color: #4338ca !important; color: white !important; }

/* Footer */
footer { margin-top: 60px; padding: 30px 0; border-top: 1px solid #e2e8f0; text-align: center; }
.footer-link { text-decoration: none; color: #475569; margin: 0 15px; font-weight: 500; }
.social-icon { width: 18px; margin-right: 8px; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

# =============================================
# 3. LOAD MODELS & CLINICAL ASSETS
# =============================================
@st.cache_resource
def load_clinical_assets():
    model = joblib.load('model_diabetes.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_clinical_assets()

HEALTHY_REF = {
    'Pregnancies': 2.0, 'Glucose': 107.0, 'BloodPressure': 70.0, 
    'SkinThickness': 27.0, 'Insulin': 102.5, 'BMI': 30.1, 
    'DiabetesPedigreeFunction': 0.33, 'Age': 27.0
}

# =============================================
# 4. SIDEBAR
# =============================================
with st.sidebar:
    st.title("DiabetesCheck AI")
    st.markdown("**Clinical Decision Support System**")
    st.caption("Version 1.1.3 - Final Build")
    st.write("---")
    st.subheader("Model Validation")
    st.metric(label="Recall (Sensitivity)", value="70.0%", help="Ability to detect positive cases.")
    st.metric(label="Accuracy Score", value="71.0%", help="Overall model precision.")
    st.write("---")
    st.info("**Engine:** XGBoost Classifier")
    st.info("**Optimization**: GridSearchCV (F1-Score)")
    st.info("**Type**: Gradient Boosted Trees.")
    st.markdown("**Source:** [Pima Indians Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)")

# =============================================
# 5. MAIN HEADER
# =============================================
if logo_base64:
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 80px; margin-right: 25px;">
        <h1 style="font-size: 2.8rem; font-weight: 700; margin: 0;">Type 2 Diabetes Risk Assessment</h1>
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("Type 2 Diabetes Risk Assessment")

st.markdown("Automated evaluation based on multivariate physiological markers.")
st.write("---")

# =============================================
# 6. CLINICAL INPUT FORM
# =============================================
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Metabolic Profile")
    glucose = st.slider("Glucose (mg/dL)", 0, 200, 115, help="2-hour plasma glucose concentration.")
    bmi = st.slider("BMI", 10.0, 50.0, 24.5, help="Body Mass Index (kg/m²).")
    insulin = st.slider("Insulin (mu U/ml)", 0, 800, 100, help="2-hour serum insulin.")
    pregnancies = st.number_input("Pregnancies", 0, 20, 0, help="Number of times pregnant.")

with col2:
    st.markdown("### Physiological Profile")
    bp = st.slider("Blood Pressure", 0, 150, 75, help="Diastolic blood pressure (mm Hg).")
    skin = st.slider("Skin Thickness", 0, 100, 25, help="Triceps skin fold thickness (mm).")
    dpf = st.slider("Pedigree Function", 0.0, 2.5, 0.47, help="Genetic risk score.")
    age = st.number_input("Age", 21, 100, 35)

st.write("##")

# =============================================
# 7. PREDICTION & ENHANCED RESULTS
# =============================================
if st.button("RUN ASSESSMENT"):
    raw_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_input = scaler.transform(raw_input)
    prob = model.predict_proba(scaled_input)[0][1]
    
    if prob < 0.35:
        risk_status, accent = "LOW RISK", "#059669"
        desc = "Statistical markers are primarily within healthy ranges."
    elif 0.35 <= prob < 0.65:
        risk_status, accent = "MODERATE RISK", "#d97706"
        desc = "Intermediate risk detected. Clinical monitoring recommended."
    else:
        risk_status, accent = "HIGH RISK", "#b91c1c"
        desc = "Strong statistical correlation with diabetic profiles."

    st.write("---")
    st.subheader("Clinical Analysis Report")
    
    # CONSOLIDATED HTML (One-liner string concatenation to fix rendering bugs)
    card_html = (
        f'<div style="background-color: #ffffff !important; padding: 30px; border-radius: 12px; '
        f'border: 1px solid #e2e8f0; border-left: 8px solid {accent}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">'
        f'<h2 style="color: {accent} !important; margin: 0; font-weight: 700; font-family: sans-serif;">{risk_status}</h2>'
        f'<p style="color: #1e293b !important; font-size: 1.25rem; margin: 15px 0 5px 0; font-family: sans-serif;">'
        f'Identified Risk Probability: <b style="font-size: 1.6rem; color: #1e293b !important;">{prob:.1%}</b></p>'
        f'<div style="background-color: #f1f5f9; border-radius: 8px; height: 16px; width: 100%; margin: 15px 0;">'
        f'<div style="background-color: {accent}; width: {prob*100}%; height: 16px; border-radius: 8px;"></div></div>'
        f'<p style="color: #475569 !important; font-weight: 500; font-size: 1.1rem; margin-bottom: 15px; font-family: sans-serif;">{desc}</p>'
        f'<p style="color: #94a3b8 !important; font-size: 0.85rem; font-style: italic; margin: 0; font-family: sans-serif;">'
        f'⚠️ Assessment based on statistical patterns. Final diagnosis must be made by a healthcare professional.</p></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # 7.3 XAI CHART
    st.write("##")
    st.subheader("Individual Factor Impact")
    st.markdown("Patient values compared against healthy population medians (1.0 = Baseline).")
    features = list(HEALTHY_REF.keys())
    ratios = [(u / r) if r != 0 else 1 for u, r in zip(raw_input[0], HEALTHY_REF.values())]
    fig = go.Figure(go.Bar(
        y=features, x=ratios, orientation='h',
        marker_color=['#b91c1c' if x > 1.25 else '#6366f1' for x in ratios],
        hovertemplate="Ratio: %{x:.2f}x<extra></extra>"
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(xaxis_title="Ratio", yaxis_autorange="reversed", template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# 8. FOOTER
# =============================================
st.markdown(f"""
<footer>
    <p style="color: #64748b; font-weight: 600;">DiabetesCheck AI | Developed by Arthur PREVEL</p>
    <div style="margin-top: 15px;">
        <a class="footer-link" href="https://www.linkedin.com/in/arthur-prével-265229332/" target="_blank">
            <img class="social-icon" src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/linkedin.svg"> LinkedIn
        </a>
        <a class="footer-link" href="https://github.com/Arthur-PREVEL" target="_blank">
            <img class="social-icon" src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg"> GitHub
        </a>
    </div>
</footer>
""", unsafe_allow_html=True)