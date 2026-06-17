import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("simple_churn_model.pkl")

st.title("Customer Churn Predictor")

st.write(
    "Predict whether a customer is likely to churn based on basic customer information."
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)

if st.button("Predict"):

    customer = pd.DataFrame([{
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Result")

    st.write(f"Churn Probability: {probability:.2%}")

    if prediction == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is likely to stay.")