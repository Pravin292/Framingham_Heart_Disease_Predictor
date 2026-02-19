import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Predict the 10-year risk of coronary heart disease using patient data.")

# Load model and scaler
model = joblib.load("model/heart_disease_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.header("🩺 Enter Patient Details")

# Input fields (ORDER MUST MATCH TRAINING DATA)
male = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
age = st.number_input("Age", min_value=20, max_value=100, value=45)
currentSmoker = st.selectbox("Current Smoker", [0, 1])
cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=70, value=0)
BPMeds = st.selectbox("On BP Medication", [0, 1])
prevalentStroke = st.selectbox("History of Stroke", [0, 1])
prevalentHyp = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
totChol = st.number_input("Total Cholesterol", value=240)
sysBP = st.number_input("Systolic BP", value=130)
diaBP = st.number_input("Diastolic BP", value=85)
BMI = st.number_input("BMI", value=25.0)
heartRate = st.number_input("Heart Rate", value=72)
glucose = st.number_input("Glucose Level", value=90)

# Predict button
if st.button("🔍 Predict Risk"):
    input_data = np.array([[
        male, age, currentSmoker, cigsPerDay, BPMeds,
        prevalentStroke, prevalentHyp, diabetes,
        totChol, sysBP, diaBP, BMI, heartRate, glucose
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease in 10 Years")
    else:
        st.success("✅ Low Risk of Heart Disease in 10 Years")