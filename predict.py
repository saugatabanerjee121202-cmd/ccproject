import joblib
import pandas as pd

# Load trained model
model = joblib.load("churn_model.pkl")

# Sample customer data
customer = {
    'SeniorCitizen': 0,
    'tenure': 12,
    'MonthlyCharges': 70,
    'TotalCharges': 840,

    'gender_Male': 1,
    'Partner_Yes': 0,
    'Dependents_Yes': 0,
    'PhoneService_Yes': 1,

    'MultipleLines_No phone service': 0,
    'MultipleLines_Yes': 0,

    'InternetService_Fiber optic': 1,
    'InternetService_No': 0,

    'OnlineSecurity_No internet service': 0,
    'OnlineSecurity_Yes': 0,

    'OnlineBackup_No internet service': 0,
    'OnlineBackup_Yes': 0,

    'DeviceProtection_No internet service': 0,
    'DeviceProtection_Yes': 0,

    'TechSupport_No internet service': 0,
    'TechSupport_Yes': 0,

    'StreamingTV_No internet service': 0,
    'StreamingTV_Yes': 1,

    'StreamingMovies_No internet service': 0,
    'StreamingMovies_Yes': 1,

    'Contract_One year': 0,
    'Contract_Two year': 0,

    'PaperlessBilling_Yes': 1,

    'PaymentMethod_Credit card (automatic)': 0,
    'PaymentMethod_Electronic check': 1,
    'PaymentMethod_Mailed check': 0
}

customer_df = pd.DataFrame([customer])

prediction = model.predict(customer_df)

probability = model.predict_proba(customer_df)[0][1]

print(f"Churn Probability: {probability:.2%}")

if prediction[0]:
    print("Prediction: Customer is likely to churn.")
else:
    print("Prediction: Customer is likely to stay.")