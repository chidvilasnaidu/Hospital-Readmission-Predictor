# 🏥 Hospital Readmission Predictor

A machine learning web application that predicts the likelihood of a diabetic patient being readmitted to the hospital based on clinical and visit information.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chidvilasnaidu/Hospital-Readmission-Predictor)
[![Python](https://img.shields.io/badge/Python-3.13.5-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://www.docker.com/)

---

## 🚀 Live Demo

👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/chidvilasnaidu/Hospital-Readmission-Predictor)**

---

## 📌 About

This app uses a trained **scikit-learn pipeline** to predict hospital readmission risk for diabetic patients. It takes patient demographics, diagnoses, and hospital visit details as input and outputs a **High Risk** or **Low Risk** prediction.

> ⚠️ This prediction is based on a ML model and is **not medical advice**.

---

## 🧠 Features

- Predicts readmission risk as **High Risk 🔴** or **Low Risk 🟢**
- Clean, responsive dark-themed UI built with Streamlit
- Input summary table shown after each prediction
- Fully containerized with Docker
- Deployed on Hugging Face Spaces

---

## 📥 Input Features

| Feature | Description |
|---|---|
| Age Group | Patient age range (e.g. [60-70)) |
| Glucose Test | Result of glucose serum test |
| A1C Test | Apolipoprotein A1C test result |
| Change in Medication | Whether diabetes medication was changed |
| Diabetes Medication | Whether diabetes medication was prescribed |
| Primary / Secondary / Additional Diagnosis | ICD-9 diagnosis categories |
| Medical Specialty | Admitting physician's specialty |
| Days in Hospital | Length of hospital stay |
| Lab Procedures | Number of lab tests performed |
| Procedures | Number of procedures during stay |
| Medications | Number of medications administered |
| Inpatient / Outpatient / Emergency Visits | Prior year visit counts |

---

## 🗂️ Project Structure

```
Hospital-Readmission-Predictor/
│
├── src/
│   ├── assets/
│   │   └── BG.png                          # Background image
│   └── streamlit_app.py                    # Main Streamlit app
│
├── hospital_readmission_pipeline.pkl       # Trained ML pipeline
├── label_encoder.pkl                       # Label encoder
├── Cleaned_data.csv                        # Cleaned dataset
├── EDA.ipynb                               # Exploratory Data Analysis
├── Hospital_Readmission_Prediction.ipynb   # Model training notebook
├── Dockerfile                              # Docker configuration
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## ⚙️ Run Locally

### With Python

```bash
git clone https://github.com/chidvilasnaidu/Hospital-Readmission-Predictor.git
cd Hospital-Readmission-Predictor
pip install -r requirements.txt
streamlit run src/streamlit_app.py
```

### With Docker

```bash
docker build -t hospital-readmission .
docker run -p 8501:8501 hospital-readmission
```

Then open **http://localhost:8501** in your browser.

---

## 🛠️ Tech Stack

- **Python 3.13.5**
- **Streamlit** — Web UI
- **scikit-learn** — ML pipeline
- **pandas** — Data handling
- **joblib** — Model serialization
- **Docker** — Containerization
- **Hugging Face Spaces** — Deployment

---

## 📊 Dataset

The model was trained on the **Diabetes 130-US Hospitals (1999–2008)** dataset, which contains over 100,000 hospital admissions for diabetic patients.

---

## 👤 Author

**Chidvilas Naidu**
- 🤗 [Hugging Face](https://huggingface.co/chidvilasnaidu)
- 💻 [GitHub](https://github.com/chidvilasnaidu)

---

## 📄 License

This project is for educational purposes only. Not intended for clinical use.
