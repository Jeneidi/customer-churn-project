import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import preprocess as prep
import joblib

def saveFinalModel():
    # 1. load & clean data
    df = prep.loadAndCleanData()
    df = df.drop(columns=["customerID"])

    # 2. split X/y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. encode
    X = pd.get_dummies(X, drop_first=True)

    # 4. train/test split
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. SMOTE oversampling
    sm = SMOTE(random_state=42)
    XTrain, yTrain = sm.fit_resample(XTrain, yTrain)

    # 6. scale numeric columns
    numericCols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    scaler = StandardScaler()
    XTrain[numericCols] = scaler.fit_transform(XTrain[numericCols])
    XTest[numericCols] = scaler.transform(XTest[numericCols])

    # 7. FINAL BEST MODEL (GBM + SMOTE)
    finalModel = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42
    )

    # 8. train final model
    finalModel.fit(XTrain, yTrain)

    # 9. save model and scaler
    joblib.dump(finalModel, "models/final_gbm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(XTrain.columns), "models/feature_columns.pkl")

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    saveFinalModel()
