import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

# --------------------------------------------------
# Streamlit Page Config (must be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="Banking Customer Churn Prediction",
    layout="centered"
)

st.title("Banking Customer Churn Prediction")

# --------------------------------------------------
# Cache model loading (CRITICAL for Streamlit Cloud)
# --------------------------------------------------
@st.cache_resource
def load_ann_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_preprocessors():
    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return label_encoder_gender, onehot_encoder_geo, scaler


# --------------------------------------------------
# Load model and preprocessors
# --------------------------------------------------
model = load_ann_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_preprocessors()

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.subheader("Enter Customer Details")

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider("Age", 18, 92, 30)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)
tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# --------------------------------------------------
# Prepare Input Data
# --------------------------------------------------
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Combine all features
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Scale features
input_data_scaled = scaler.transform(input_data)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled, verbose=0)
    churn_probability = prediction[0][0]

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{churn_probability:.2f}**")

    if churn_probability > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
