import joblib

df = joblib.load("models/tzeltal/dataset.pkl")
print(df.columns)
