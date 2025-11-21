import pandas as pd
import joblib

# Load all necessary components
model = joblib.load("models/final_gbm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
featureColumns = joblib.load("models/feature_columns.pkl")

def predictCustomerChurn(customerDict):

    # Convert the dictionary into a DataFrame with one row
    df = pd.DataFrame([customerDict])

    # One-hot encode exactly like training
    df = pd.get_dummies(df, drop_first=True)

    # Add any missing columns (because training had more categories)
    for col in featureColumns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column ordering
    df = df[featureColumns]

    # Scale numeric columns
    numericCols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    df[numericCols] = scaler.transform(df[numericCols])

    # Predict churn
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability

if __name__ == "__main__":
    # Example input for testing
    sampleCustomer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.5,
        "TotalCharges": 985.2
    }

    pred, prob = predictCustomerChurn(sampleCustomer)
    print("\nPrediction:", pred)
    print("Churn Probability:", round(prob, 4))
