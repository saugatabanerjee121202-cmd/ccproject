import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd

# Load model
model = joblib.load("simple_churn_model.pkl")


def predict_churn():
    try:
        senior = int(senior_var.get())
        tenure = float(tenure_entry.get())
        monthly = float(monthly_entry.get())
        total = float(total_entry.get())

        customer = pd.DataFrame([{
            "SeniorCitizen": senior,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        prediction = model.predict(customer)[0]
        probability = model.predict_proba(customer)[0][1]

        if prediction == 1:
            result = (
                f"Churn Probability: {probability:.2%}\n\n"
                "Customer is likely to churn."
            )
        else:
            result = (
                f"Churn Probability: {probability:.2%}\n\n"
                "Customer is likely to stay."
            )

        messagebox.showinfo("Prediction Result", result)

    except ValueError:
        messagebox.showerror(
            "Input Error",
            "Please enter valid numeric values."
        )


root = tk.Tk()
root.title("Customer Churn Predictor")
root.geometry("400x300")

tk.Label(root, text="Customer Churn Predictor",
         font=("Arial", 14, "bold")).pack(pady=10)

tk.Label(root, text="Senior Citizen (0 = No, 1 = Yes)").pack()
senior_var = tk.StringVar(value="0")
tk.Entry(root, textvariable=senior_var).pack()

tk.Label(root, text="Tenure (months)").pack()
tenure_entry = tk.Entry(root)
tenure_entry.pack()

tk.Label(root, text="Monthly Charges").pack()
monthly_entry = tk.Entry(root)
monthly_entry.pack()

tk.Label(root, text="Total Charges").pack()
total_entry = tk.Entry(root)
total_entry.pack()

tk.Button(
    root,
    text="Predict",
    command=predict_churn
).pack(pady=15)

root.mainloop()