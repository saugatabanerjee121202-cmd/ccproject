import pandas as pd

df = pd.read_csv("dataset.csv")

df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.dropna()

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn_Yes", axis=1)

print(X.columns.tolist())