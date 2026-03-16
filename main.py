import streamlit as st
import base64
import joblib
import numpy as np
import pandas as pd

# PAGE CONFIG (must be first)
st.set_page_config(page_title="Medical Prediction App", layout="wide")

# ---------- LOAD BACKGROUND ----------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("IMGs/BG.png")

# ---------- LOAD MODEL & LABEL ENCODER ----------
@st.cache_resource
def load_model():
    pipeline = joblib.load("hospital_readmission_pipeline.pkl")
    le       = joblib.load("label_encoder.pkl")
    return pipeline, le

pipeline, le = load_model()

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

    # Validate all dropdowns are filled
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
        # Build a single-row DataFrame matching the training column order
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

        # Run through the full pipeline (encoding + scaling + model)
        prediction_encoded = pipeline.predict(input_df)[0]

       
        # Decode label (0/1 -> yes/no)
        prediction_label = le.inverse_transform([prediction_encoded])[0]

        # Show result
        st.divider()
        if prediction_label == "yes":
            st.error("🔴 **High Risk — Patient is likely to be readmitted**")
        else:
            st.success("🟢 **Low Risk — Patient is unlikely to be readmitted**")


        # Show input summa
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


# ---------- STYLING ----------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    * {{ box-sizing: border-box; }}
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stHeader"], div[data-testid="stToolbar"], div[data-testid="stDecoration"] {{ display: none !important; }}

    .stApp {{
        background: linear-gradient(rgba(2,8,20,0.78), rgba(2,8,20,0.82)), url("data:image/png;base64,{img}");
        background-size: cover; background-position: center; background-repeat: no-repeat;
        font-family: 'DM Sans', sans-serif;
    }}
    .block-container {{ padding-top: 110px !important; padding-bottom: 120px !important; max-width: 95% !important; padding-left: 40px !important; padding-right: 40px !important; }}

    h1, h2, h3, h4 {{ color: white !important; font-family: 'DM Sans', sans-serif !important; letter-spacing: -0.3px; }}
    h1 {{ font-size: 4rem !important; font-weight: 700 !important; }}
    h3 {{ font-size: 2.1rem !important; font-weight: 600 !important; color: #94d4f5 !important; text-transform: uppercase; letter-spacing: 1px; }}
    p, span, div, label {{ color: rgba(255,255,255,0.85) !important; font-family: 'DM Sans', sans-serif !important; }}
    .stMarkdown p {{ color: rgba(255,255,255,0.6) !important; font-size: 0.95rem !important; }}
    hr {{ border-color: rgba(255,255,255,0.08) !important; margin: 20px 0 !important; }}

    [data-testid="stWidgetLabel"] label, .stNumberInput label, .stSelectbox label, [data-testid="stWidgetLabel"] p {{
        color: rgba(255,255,255,0.70) !important; font-size: 13px !important; font-weight: 500 !important;
        letter-spacing: 0.2px !important; margin-bottom: 4px !important;
    }}
    .stNumberInput, .stNumberInput > div {{ width: 100% !important; }}
    div[data-baseweb="input"] {{ background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.13) !important; border-radius: 8px !important; height: 42px !important; width: 100% !important; }}
    div[data-baseweb="input"] input {{ background: transparent !important; border: none !important; color: white !important; font-size: 14px !important; height: 40px !important; font-family: 'DM Sans', sans-serif !important; }}
    div[data-baseweb="input"]:focus-within {{ border-color: rgba(56,189,248,0.5) !important; box-shadow: 0 0 0 2px rgba(56,189,248,0.08) !important; }}
    div[data-baseweb="input"] button {{ color: rgba(255,255,255,0.5) !important; background: transparent !important; border: none !important; }}
    div[data-baseweb="input"] button:hover {{ color: #38bdf8 !important; background: rgba(56,189,248,0.1) !important; }}

    .stSelectbox, .stSelectbox > div, div[data-baseweb="select"] {{ width: 100% !important; }}
    div[data-baseweb="select"] > div {{ background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.13) !important; border-radius: 8px !important; height: 42px !important; min-height: 42px !important; display: flex !important; align-items: center !important; width: 100% !important; }}
    div[data-baseweb="select"] > div:focus-within {{ border-color: rgba(56,189,248,0.5) !important; box-shadow: 0 0 0 2px rgba(56,189,248,0.08) !important; }}
    div[data-baseweb="select"] span, div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {{ color: white !important; font-size: 14px !important; font-family: 'DM Sans', sans-serif !important; }}
    div[data-baseweb="select"] svg {{ fill: rgba(255,255,255,0.5) !important; }}

    [data-baseweb="popover"] ul {{ background: #0b1a2e !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 8px !important; padding: 4px !important; }}
    [data-baseweb="popover"] li {{ color: rgba(255,255,255,0.85) !important; font-size: 13.5px !important; border-radius: 6px !important; padding: 8px 12px !important; }}
    [data-baseweb="popover"] li:hover {{ background: rgba(56,189,248,0.15) !important; color: white !important; }}

    [data-testid="column"] {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 24px 28px !important; backdrop-filter: blur(8px); }}

    div[data-testid="stButton"] > button {{ display: block; margin: 0 auto; background: linear-gradient(135deg, #0ea5e9, #0284c7); color: white !important; font-family: 'DM Sans', sans-serif !important; font-size: 15px !important; font-weight: 600 !important; letter-spacing: 0.5px; padding: 12px 48px !important; border: none !important; border-radius: 10px !important; cursor: pointer; transition: background 0.3s ease, transform 0.15s ease, box-shadow 0.3s ease !important; box-shadow: 0 4px 20px rgba(14,165,233,0.3); }}
    div[data-testid="stButton"] > button:hover {{ background: linear-gradient(135deg, #22c55e, #16a34a) !important; box-shadow: 0 6px 28px rgba(34,197,94,0.45) !important; transform: translateY(-2px) !important; color: white !important; }}
    div[data-testid="stButton"] > button:active {{ transform: translateY(0px) !important; box-shadow: 0 2px 12px rgba(34,197,94,0.3) !important; }}

    [data-testid="stAlert"] {{ background: rgba(34,197,94,0.12) !important; border: 1px solid rgba(34,197,94,0.35) !important; border-radius: 10px !important; color: white !important; }}

    /* ── Expander ── */
    [data-testid="stExpander"] {{ background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.10) !important; border-radius: 10px !important; }}
    [data-testid="stExpander"] summary {{ color: rgba(255,255,255,0.75) !important; font-size: 13.5px !important; }}
    [data-testid="stExpander"] svg {{ fill: rgba(255,255,255,0.5) !important; }}

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"], [data-testid="stDataFrame"] > div, .stDataFrame iframe {{ background: transparent !important; }}
    [data-testid="stDataFrame"] div[data-testid="stDataFrameResizable"] {{ background: rgba(10,20,40,0.85) !important; border-radius: 8px !important; border: 1px solid rgba(255,255,255,0.10) !important; }}
    .dvn-scroller {{ background: rgba(10,20,40,0.85) !important; }}
    .glideDataEditor {{ background: rgba(10,20,40,0.85) !important; }}
    canvas {{ filter: invert(0) !important; }}

    .site-header {{ position: fixed; top: 0; left: 0; width: 100%; z-index: 9999; display: flex; justify-content: space-between; align-items: center; padding: 14px 48px; background: rgba(2, 10, 24, 0.85); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border-bottom: 1px solid rgba(255,255,255,0.08); }}
    .logo {{ font-family: 'Space Mono', monospace; font-size: 20px; font-weight: 700; color: white !important; letter-spacing: -0.5px; }}
    .logo span {{ color: #38bdf8 !important; }}
    .site-header nav a {{ margin-left: 28px; color: rgba(255,255,255,0.65) !important; text-decoration: none; font-size: 13.5px; font-weight: 500; letter-spacing: 0.2px; transition: color 0.2s ease; }}
    .site-header nav a:hover {{ color: #38bdf8 !important; }}

    .site-footer {{ position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; padding: 10px; background: rgba(2, 10, 24, 0.85); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border-top: 1px solid rgba(255,255,255,0.07); color: rgba(255,255,255,0.4) !important; font-size: 12.5px; }}
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