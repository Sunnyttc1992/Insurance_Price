from operator import le
from unicodedata import numeric
from matplotlib.pyplot import sca
from sklearn.feature_selection import chi2
import streamlit as st
import pandas as pd
import numpy as np
import joblib


scaler = joblib.load("/workspaces/Insurance_Price/label/scaler.pkl")
le_sex = joblib.load("/workspaces/Insurance_Price/label/label_encoder_sex.joblib")
le_smoker = joblib.load("/workspaces/Insurance_Price/label/label_encoder_smoker.joblib")
le_region = joblib.load("/workspaces/Insurance_Price/label/label_encoder_region.joblib")
le_bmi_cat = joblib.load("/workspaces/Insurance_Price/label/label_encoder_bmi_category.joblib")
le_age_group = joblib.load("/workspaces/Insurance_Price/label/label_encoder_age_group.joblib")

model = joblib.load("/workspaces/Insurance_Price/model/best_model.pkl")

st.set_page_config(page_title = "Insurance Claim Predictor",layout ="wide")
st.title("Health Insurance Payment Prediction App")
st.write("Please enter the detail to estimate your insurance amount.")

with st.form("input_form"):
    col1,col2 = st.columns(2)
    with col1:
        age = st.number_input("Age",min_value = 18 ,max_value = 100,value = 30)
        bmi = st.number_input("BMI",min_value = 10.0 ,max_value = 50.0,value = 25.0)
        children = st.number_input("Children",min_value = 0 ,max_value = 10,value = 0)
        bmi_category = st.selectbox("BMI Category",options =le_bmi_cat.classes_)
    with col2:
        sex = st.selectbox("Sex",options =le_sex.classes_)
        smoker = st.selectbox("Smoker",options =le_smoker.classes_)
        region = st.selectbox("Region",options =le_region.classes_)
        age_group = st.selectbox("Age Group",options =le_age_group.classes_)


    submit_button = st.form_submit_button(label ="Predict Insurance Charge")


if submit_button:

    input_data = pd.DataFrame ({
        "age":[age],
        "sex":[le_sex.transform([sex])[0]],
        "bmi":[bmi],
        "children":[children],
        "smoker":[le_smoker.transform([smoker])[0]],    
        "region":[le_region.transform([region])[0]],
        "age_group":[le_age_group.transform([age_group])[0]],
        "bmi_category":[le_bmi_cat.transform([bmi_category])[0]]
    })

    numeric_features = ["age","bmi","children"]
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])

    prediction = model.predict(input_data)[0]

    st.success(f"The estimated insurance charge is: ${prediction:.2f}")
