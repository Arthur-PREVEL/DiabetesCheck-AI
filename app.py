"""
DiabetesCheck AI - Clinical Decision Support System (DSS)
---------------------------------------------------------
Author: Arthur PREVEL
Description: Streamlit application for Type 2 Diabetes risk assessment using 
             an XGBoost classifier and interactive XAI visualizations.
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import base64

# =============================================
# 1. PAGE CONFIGURATION & HELPER FUNCTIONS
# =============================================

st.set_page_config(
    page_title="DiabetesCheck AI - Clinical DSS",
    page_icon="images/favicon.ico",  # Updated path to images folder
    layout="wide"
)

def get_base64_image(image_path):
    """
    Reads a local image file and encodes it into a base64 string
    for embedding directly into HTML containers.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Load and encode the brand logo image
logo_base64 = get_base64_image("images/logo.png") # Updated path

# =============================================
# 2. CUSTOM CSS STYLING
# =============================================

st.markdown("""
    <style>
    /* Import clinical-grade font Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1e293b; }
    
    /* --- SIDEBAR: High Contrast Fix --- */
    [data-testid="stSidebar"] { background-color: #f1f5f9 !important; }
    [data-testid="stSidebar"] * { color: #1e293b !important; }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        padding: 15px !important;
        border-radius: 10px !important;
    }
    [data-testid="stMetricLabel"] { color: #475569 !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 700 !important; }
    
    /* Assessment Result Card Container */
    .report-card { 
        background-color: white; padding: 2rem; border-radius: 12px; 
        border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 25px;
    }
    
    /* Primary Action Button */
    .stButton>button {
        width: 100%; border-radius: 8px; background-color: #1e1b4b;
        color: white !important; font-weight: 600; height: 3.5rem; border: none;
    }
    .stButton>button:hover { background-color: #4338ca !important; border: none; color: white !important; }
    
    /* Footer & Disclaimer Styling */
    footer { margin-top: 80px; padding: 40px 0; border-top: 1px solid #e2e8f0; text-align: center; color: #64748b; }
    .footer-link { text-decoration: none; color: #475569; margin: 0 15px; font-weight: 500; }
    .social-icon { width: 20px; margin-right: 8px; vertical-align: middle; filter: grayscale(100%); transition: 0.3s; }
    .footer-link:hover .social-icon { filter: grayscale(0%); }
    .disclaimer-text { font-size: 0.85rem; color: #94a3b8; font-style: italic; margin-top: 10px; }

    /* --- BRAND HEADER FLEXBOX --- */
    .brand-header {
        display: flex;
        align-items: center;
        padding-bottom: 20px;
    }
    .brand-logo {
        width: 80px;
        height: auto;
        margin-right: 25px;
    }
    .brand-title {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================
# 3. LOAD MODELS & CLINICAL ASSETS
# =============================================

@st.cache_resource
def load_clinical_assets():
    """
    Loads the serialized XGBoost model and the Standard Scaler.
    Uses caching to prevent reloading on every user interaction.
    """
    model = joblib.load('model_diabetes.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_clinical_assets()

# Reference medians defined from the healthy population segment of the dataset.
# Used as a baseline (1.0 ratio) for the XAI feature contribution chart.
HEALTHY_REF = {
    'Pregnancies': 2.0, 'Glucose': 107.0, 'BloodPressure': 70.0, 
    'SkinThickness': 27.0, 'Insulin': 102.5, 'BMI': 30.1, 
    'DiabetesPedigreeFunction': 0.33, 'Age': 27.0
}

# =============================================
# 4. SIDEBAR: METRICS & TRANSPARENCY
# =============================================
with st.sidebar:
    st.title("DiabetesCheck AI")
    st.markdown("**Clinical Decision Support System**")
    st.write("---")
    
    st.subheader("Model Validation")
    st.metric(label="Recall (Sensitivity)", value="70.0%", help="Model's ability to identify diabetic patients accurately.")
    st.metric(label="Accuracy Score", value="71.0%", help="General precision of the XGBoost model on the test set.")
    
    st.write("---")
    st.subheader("Algorithm Transparency")
    st.info("""
        **Engine:** XGBoost Classifier  
        **Optimization:** GridSearchCV (F1-Score)  
        **Type:** Gradient Boosted Trees.
    """)
    st.write("---")
    st.markdown("**Source:** [Pima Indians Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)")

# =============================================
# 5. MAIN INTERFACE & BRANDING
# =============================================

# Render the brand header using HTML/CSS for perfect alignment
if logo_base64:
    st.markdown(f"""
        <div class="brand-header">
            <img src="data:image/png;base64,{logo_base64}" class="brand-logo" alt="DiabetesCheck AI Logo">
            <h1 class="brand-title">Type 2 Diabetes Risk Assessment</h1>
        </div>
        """, unsafe_allow_html=True)
else:
    # Fallback title if logo image fails to load
    st.title("Type 2 Diabetes Risk Assessment")

st.markdown("Automated evaluation based on multivariate physiological markers.")
st.write("---")

# =============================================
# 6. CLINICAL INPUT FORM
# =============================================
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Metabolic Profile")
        glucose = st.slider("Glucose Concentration (mg/dL)", 0, 200, 115, 
                            help="2-hour plasma glucose concentration in an oral glucose tolerance test.")
        bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 24.5, 
                        help="Weight in kg/(height in m)². A key indicator of body fatness.")
        insulin = st.slider("Serum Insulin (mu U/ml)", 0, 800, 100, 
                            help="2-hour serum insulin concentration (mu U/ml).")
        pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0, 
                                      help="Number of times pregnant.")
    with col2:
        st.markdown("### Physiological Profile")
        bp = st.slider("Blood Pressure (mm Hg)", 0, 150, 75, 
                        help="Diastolic blood pressure (mm Hg).")
        skin = st.slider("Skin Thickness (mm)", 0, 100, 25, 
                          help="Triceps skin fold thickness used to estimate body fat.")
        dpf = st.slider("Pedigree Function", 0.0, 2.5, 0.47, 
                         help="Genetic risk score based on family history of diabetes.")
        age = st.number_input("Patient Age", 21, 100, 35, 
                               help="Patient age in years.")

st.write("##")

# =============================================
# 7. PREDICTION LOGIC & XAI VISUALIZATION
# =============================================
if st.button("RUN ASSESSMENT"):
    # 7.1 Data Preparation & Prediction
    raw_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_input = scaler.transform(raw_input)
    prob = model.predict_proba(scaled_input)[0][1]
    prediction = model.predict(scaled_input)

    st.write("---")
    st.subheader("Clinical Analysis Report")
    
    # Determine status and color accent based on prediction
    status = "ELEVATED RISK" if prediction[0] == 1 else "LOW RISK"
    accent = "#b91c1c" if prediction[0] == 1 else "#059669"
    
    # 7.2 Render Results Card
    st.markdown(f"""
        <div class="report-card" style="border-left: 6px solid {accent};">
            <h2 style="color: {accent}; margin: 0;">{status}</h2>
            <p style="font-size: 1.1rem; margin-top: 10px;">
                Identified statistical probability: <b>{prob:.1%}</b>.
            </p>
            <div style="background-color: #f1f5f9; border-radius: 8px; height: 12px; margin: 1rem 0;">
                <div style="background-color: {accent}; width: {prob*100}%; height: 12px; border-radius: 8px;"></div>
            </div>
            <p class="disclaimer-text">
                ⚠️ Assessment based on statistical patterns. Always consult a medical professional for diagnosis.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 7.3 Interactive Feature Contribution Chart (XAI)
    st.write("##")
    st.subheader("Individual Factor Impact")
    st.markdown("Patient values compared against healthy population medians (1.0 = Baseline).")
    
    features = list(HEALTHY_REF.keys())
    # Calculate ratios for the chart against healthy baseline
    ratios = [(u / r) if r != 0 else 1 for u, r in zip(raw_input[0], HEALTHY_REF.values())]
    
    fig = go.Figure(go.Bar(
        y=features, x=ratios, orientation='h',
        # Color bar red if ratio exceeds 1.25x baseline
        marker_color=['#b91c1c' if x > 1.25 else '#6366f1' for x in ratios],
        hovertemplate="Ratio: %{x:.2f}x<extra></extra>"
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(xaxis_title="Ratio", yaxis_autorange="reversed", 
                      template="plotly_white", height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# 8. PROFESSIONAL FOOTER
# =============================================
st.markdown(f"""
    <footer>
        <p><b>DiabetesCheck AI</b> | Developed by <b>Arthur PREVEL</b></p>
        <div style="margin-top: 15px;">
            <a class="footer-link" href="https://www.linkedin.com/in/arthur-prével-265229332/" target="_blank">
                <img class="social-icon" src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/linkedin.svg"> LinkedIn
            </a>
            <a class="footer-link" href="https://github.com/Arthur-PREVEL" target="_blank">
                <img class="social-icon" src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg"> GitHub
            </a>
        </div>
        <p class="disclaimer-text">
            Disclaimer: This tool is for educational purposes only. Always consult a medical professional if needed.<br>
            Consulter un médecin au besoin.
        </p>
    </footer>
    """, unsafe_allow_html=True)