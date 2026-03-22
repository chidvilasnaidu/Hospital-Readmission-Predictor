import streamlit as st
import joblib
import numpy as np
import pandas as pd


st.set_page_config(page_title="Medical Prediction App", layout="wide")


@st.cache_resource
def load_model():
    pipeline = joblib.load("hospital_readmission_pipeline.pkl")
    le       = joblib.load("label_encoder.pkl")
    return pipeline, le

pipeline, le = load_model()

st.title("Hospital Readmission Prediction")
st.write("Predict the likelihood of patient readmission based on clinical information.")
st.divider()

col1, col2 = st.columns(2, gap="large")


with col1:
    st.subheader("Patient Information")
    age           = st.selectbox("Enter Age Group",                           ["", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    glucose_test  = st.selectbox("Glucose Test",["", "normal", "high", "no"])
    A1Ctest       = st.selectbox("A1C Test (Apolipoprotein Test)",            ["", "normal", "high", "no"])
    change        = st.selectbox("Change in diabetes medication?",            ["", "yes", "no"])
    diabetes_med  = st.selectbox("Diabetes medication prescribed?",           ["", "yes", "no"])
    diag_1        = st.selectbox("Primary Diagnosis",                         ["", "Circulatory", "Respiratory", "Digestive", "Other", "Diabetes", "Injury", "Musculoskeletal", "Missing"])
    diag_2        = st.selectbox("Secondary Diagnosis",                       ["", "Circulatory", "Respiratory", "Digestive", "Other", "Diabetes", "Injury", "Musculoskeletal", "Missing"])
    diag_3        = st.selectbox("Additional Secondary Diagnosis",            ["", "Circulatory", "Respiratory", "Digestive", "Other", "Diabetes", "Injury", "Musculoskeletal", "Missing"])

with col2:
    st.subheader("Hospital Visit Details")
    medical_specialty = st.selectbox(
        "Medical Specialty",
        ["", "Not specified", "Cardiology", "Surgery", "InternalMedicine", "Other", "Emergency/Trauma", "Family/GeneralPractice"]
    )
    time_in_hospital  = st.number_input("Days spent in hospital",  min_value=1,  max_value=14, value=1)
    n_lab_procedures  = st.number_input("Number of laboratory procedures",min_value=0,  value=0)
    n_procedures      = st.number_input("Number of procedures during stay",   min_value=0,  value=0)
    n_medications     = st.number_input("Number of medications administered", min_value=0,  value=0)
    n_inpatient       = st.number_input("Inpatient visits last year",         min_value=0,  value=0)
    n_outpatient      = st.number_input("Outpatient visits last year",        min_value=0,  value=0)
    n_emergency       = st.number_input("Emergency visits last year",         min_value=0,  value=0)

st.divider()


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
"""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

* { box-sizing: border-box; }

#MainMenu, footer { visibility: hidden; }

[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"] { display: none !important; }

.stApp {
    background: #000000;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 110px !important;
    padding-bottom: 120px !important;
    max-width: 95% !important;
    padding-left: 40px !important;
    padding-right: 40px !important;
}

h1, h2, h3, h4 {
    color: white !important;
}

.site-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 9999;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 48px;
    background: rgba(2, 10, 24, 0.85);
    backdrop-filter: blur(16px);
}

.logo {
    font-family: 'Space Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: white !important;
}

.logo span {
    color: #38bdf8 !important;
}

.site-header nav a {
    margin-left: 28px;
    color: rgba(255,255,255,0.65) !important;
    text-decoration: none;
}

.site-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    background: rgba(2, 10, 24, 0.85);
    backdrop-filter: blur(16px);
    color: rgba(255,255,255,0.4) !important;
    font-size: 12.5px;
}

</style>

<div class="site-header">
<div class="logo">Med<span>Portal</span></div>
<nav>
<a href="#">Home</a>
<a href="#">About</a>
<a href="#">Services</a>
<a href="#">Contact</a>
</nav>
</div>

<div class="site-footer">
This prediction is based on a ML model and is not medical advice. Please consult a healthcare professional or contact a hospital in case of emergency.
</div>
""",
unsafe_allow_html=True
)
