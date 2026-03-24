import os
import base64
import pandas as pd
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(base_dir, "hospital_readmission_pipeline.pkl")
    encoder_path  = os.path.join(base_dir, "label_encoder.pkl")
    pipeline = joblib.load(pipeline_path)
    le       = joblib.load(encoder_path)
    return pipeline, le

pipeline, le = load_model()

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_path = "IMGs\BG.png"
b64 = get_base64_image(img_path)

# ---------- UI ----------
st.title("Hospital Readmission Prediction")
st.write("Predict the likelihood of patient readmission based on clinical information.")
st.divider()

col1, col2 = st.columns(2, gap="large")

# LEFT SIDE
with col1:
    st.subheader("Patient Information")
    age           = st.selectbox("Enter Age Group",                           ["", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    glucose_test  = st.selectbox("Glucose Test",                              ["", "normal", "high", "no"])
    A1Ctest       = st.selectbox("A1C Test (Apolipoprotein Test)",            ["", "normal", "high", "no"])
    change        = st.selectbox("Change in diabetes medication?",            ["", "yes", "no"])
    diabetes_med  = st.selectbox("Diabetes medication prescribed?",           ["", "yes", "no"])
    diag_1        = st.selectbox("Primary Diagnosis",                         ["", "Circulatory", "Respiratory", "Digestive", "Other", "Diabetes", "Injury", "Musculoskeletal", "Missing"])
    diag_2        = st.selectbox("Secondary Diagnosis",                       ["", "Circulatory", "Respiratory", "Digestive", "Other", "Diabetes", "Injury", "Musculoskeletal", "Missing"])
    diag_3        = st.selectbox("Additional Secondary Diagnosis",            ["", "Circulatory", "Respiratory", "Digestive", "Other", "Diabetes", "Injury", "Musculoskeletal", "Missing"])

# RIGHT SIDE
with col2:
    st.subheader("Hospital Visit Details")
    medical_specialty = st.selectbox(
        "Medical Specialty",
        ["", "Not specified", "Cardiology", "Surgery", "InternalMedicine", "Other", "Emergency/Trauma", "Family/GeneralPractice"]
    )
    time_in_hospital  = st.number_input("Days spent in hospital",             min_value=1,  max_value=14, value=1)
    n_lab_procedures  = st.number_input("Number of laboratory procedures",    min_value=0,  value=0)
    n_procedures      = st.number_input("Number of procedures during stay",   min_value=0,  value=0)
    n_medications     = st.number_input("Number of medications administered", min_value=0,  value=0)
    n_inpatient       = st.number_input("Inpatient visits last year",         min_value=0,  value=0)
    n_outpatient      = st.number_input("Outpatient visits last year",        min_value=0,  value=0)
    n_emergency       = st.number_input("Emergency visits last year",         min_value=0,  value=0)

st.divider()

# ---------- PREDICT ----------
if st.button("Predict Readmission Risk"):

    required_fields = {
        "Age Group": age,
        "Glucose Test": glucose_test,
        "A1C Test": A1Ctest,
        "Change in medication": change,
        "Diabetes medication": diabetes_med,
        "Primary Diagnosis": diag_1,
        "Secondary Diagnosis": diag_2,
        "Additional Diagnosis": diag_3,
        "Medical Specialty": medical_specialty,
    }
    missing = [k for k, v in required_fields.items() if v == ""]

    if missing:
        st.warning(f"Please fill in: {', '.join(missing)}")
    else:
        input_df = pd.DataFrame([{
            "age":               age,
            "time_in_hospital":  int(time_in_hospital),
            "n_lab_procedures":  int(n_lab_procedures),
            "n_procedures":      int(n_procedures),
            "n_medications":     int(n_medications),
            "n_outpatient":      int(n_outpatient),
            "n_inpatient":       int(n_inpatient),
            "n_emergency":       int(n_emergency),
            "medical_specialty": medical_specialty,
            "diag_1":            diag_1,
            "diag_2":            diag_2,
            "diag_3":            diag_3,
            "glucose_test":      glucose_test,
            "A1Ctest":           A1Ctest,
            "change":            change,
            "diabetes_med":      diabetes_med,
        }])

        prediction_encoded = pipeline.predict(input_df)[0]
        prediction_label = le.inverse_transform([prediction_encoded])[0]

        st.divider()
        if prediction_label == "yes":
            st.error("🔴 **High Risk — Patient is likely to be readmitted**")
        else:
            st.success("🟢 **Low Risk — Patient is unlikely to be readmitted**")

        with st.expander("View Input Summary"):
            rows = ""
            for col, val in input_df.iloc[0].items():
                label = col.replace("_", " ").title()
                rows += (
                    "<tr>"
                    f"<td style='padding:7px 16px;color:rgba(255,255,255,0.55);font-size:13px;border-bottom:1px solid rgba(255,255,255,0.06);'>{label}</td>"
                    f"<td style='padding:7px 16px;color:white;font-size:13px;font-weight:500;border-bottom:1px solid rgba(255,255,255,0.06);'>{val}</td>"
                    "</tr>"
                )
            st.markdown(
                "<table style='width:100%;border-collapse:collapse;background:rgba(255,255,255,0.04);border-radius:8px;overflow:hidden;'>"
                "<thead><tr>"
                "<th style='padding:8px 16px;text-align:left;color:#94d4f5;font-size:12px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid rgba(255,255,255,0.12);'>Field</th>"
                "<th style='padding:8px 16px;text-align:left;color:#94d4f5;font-size:12px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid rgba(255,255,255,0.12);'>Value</th>"
                f"</tr></thead><tbody>{rows}</tbody></table>",
                unsafe_allow_html=True,
            )

st.markdown(
f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
* {{ box-sizing: border-box; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"] {{ display: none !important; }}
.stApp {{
    background: linear-gradient(rgba(2,8,20,0.78), rgba(2,8,20,0.82)),
                url("data:image/png;base64,{b64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    font-family: 'DM Sans', sans-serif;
}}
.block-container {{
    padding-top: 55px !important;
    padding-bottom: 80px !important;
    max-width: 92% !important;
    padding-left: 36px !important;
    padding-right: 36px !important;
}}
h1 {{
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 2.3rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    margin-top: 0 !important;
    margin-bottom: 4px !important;
    padding-top: 0 !important;
}}
.stApp p {{
    font-size: 0.85rem !important;
    color: rgba(255,255,255,0.75) !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
}}
hr {{
    border-color: rgba(255,255,255,0.08) !important;
    margin: 10px 0 !important;
}}
h3, [data-testid="column"] h3,
[data-testid="stVerticalBlock"] h3,
.stApp h3, div h3 {{
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-top: 0 !important;
    margin-bottom: 8px !important;
    opacity: 1 !important;
    visibility: visible !important;
}}
h3 a {{ display: none !important; }}
[data-testid="stVerticalBlock"] > div {{
    gap: 6px !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    gap: 6px !important;
}}
[data-testid="column"] {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    backdrop-filter: blur(8px) !important;
}}
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {{
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    margin-bottom: 3px !important;
}}
div[data-baseweb="input"] {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 7px !important;
    height: 38px !important;
}}
div[data-baseweb="input"] input {{
    color: white !important;
    font-size: 0.82rem !important;
    padding: 0 10px !important;
}}
div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 7px !important;
    min-height: 38px !important;
    padding: 2px 8px !important;
}}
div[data-baseweb="select"] span {{
    color: white !important;
    font-size: 0.82rem !important;
}}
div[data-baseweb="select"] svg {{
    fill: rgba(255,255,255,0.6) !important;
}}
ul[data-baseweb="menu"] {{
    background: #0d1b2e !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}}
li[role="option"] {{
    color: white !important;
    font-size: 0.82rem !important;
}}
li[role="option"]:hover {{
    background: rgba(14,165,233,0.2) !important;
}}
div[data-testid="stButton"] > button {{
    display: block;
    margin: 0 auto;
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 9px 36px !important;
    border-radius: 8px !important;
    border: none !important;
}}
div[data-testid="stButton"] > button:hover {{
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
}}
/* ── Header ── */
.site-header {{
    position: fixed;
    top: 0; left: 0;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 36px;
    background: rgba(2,10,24,0.95) !important;
    backdrop-filter: blur(8px);
    z-index: 9999 !important;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.site-header * {{
    visibility: visible !important;
    opacity: 1 !important;
}}
.logo {{
    font-family: 'Space Mono', monospace !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: white !important;
}}
.logo span {{
    color: #38bdf8 !important;
}}
/* ── Nav links ── */
.site-header nav {{
    display: flex;
    gap: 24px;
    align-items: center;
}}
.site-header nav a {{
    color: rgba(255,255,255,0.65) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-decoration: none !important;
    transition: color 0.2s;
}}
.site-header nav a:hover {{
    color: #38bdf8 !important;
}}
/* ── Footer ── */
.site-footer {{
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 8px;
    background: rgba(2,10,24,0.90);
    color: rgba(255,255,255,0.35) !important;
    font-size: 11px;
    border-top: 1px solid rgba(255,255,255,0.05);
}}
</style>
<div class="site-header">
    <div class="logo">Med<span>Portal</span></div>
    <nav>
        <a href="#">Home</a>
        <a href="https://github.com/chidvilasnaidu/Hospital-Readmission-Predictor/blob/main/README.md">About</a>
        <a href="#">Services</a>
        <a href="http://www.linkedin.com/in/chidvilas-kumkapalla-60703a1b3">Contact</a>
    </nav>
</div>
<div class="site-footer">
    This prediction is based on a ML model and is not medical advice.
</div>
""",
unsafe_allow_html=True)



