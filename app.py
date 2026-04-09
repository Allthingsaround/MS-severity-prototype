import numpy as np
import joblib
import streamlit as st

model = joblib.load("ms_severity_model.pkl")

st.title("Multiple Sclerosis Severity Prediction Tool")
st.write("Enter EDSS score and MRI lesion metrics.")

EDSS = st.slider("EDSS score", 0.0, 10.0, step=0.5)

Lesion_volume = st.number_input("Lesion Volume (mL)", min_value=0.000)
Mean_lesion = st.number_input("Mean Lesion Size (mL)", min_value=0.000)
Lesion_count = st.number_input("Lesion Count", min_value=0)

if st.button("Predict Severity"):
    features = np.array([[Lesion_volume, EDSS, Mean_lesion, Lesion_count]])
    pred = model.predict(features)[0]
    
    severity_lvl = {
        0: "Mild",
        1: "Moderate",
        2: "Severe"
    }
    
    st.subheader("Predicted Severity Stage: ")
    st.success(severity_lvl[pred])
