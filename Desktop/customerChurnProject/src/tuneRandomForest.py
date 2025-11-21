import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import preprocess as prep
import numpy as np

def tuneRandomForest():
    # 1. load and clean data
    df = prep.loadAndCleanData()
    df = df.drop(columns=["customerID"])

    # 2. separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # 4. split before SMOTE
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. apply SMOTE
    sm = SMOTE(random_state=42)
    XTrain, yTrain = sm.fit_resample(XTrain, yTrain)

    # 6. scale numeric features
    numericCols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    scaler = StandardScaler()
    XTrain[numericCols] = scaler.fit_transform(XTrain[numericCols])
    XTest[numericCols] = scaler.transform(XTest[numericCols])

    # 7. define search space for Random Forest
    paramGrid = {
        "n_estimators": [100, 200, 300, 400, 500, 600],
        "max_depth": [None, 5, 10, 15, 20, 25, 30],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["auto", "sqrt", "log2"]
    }

    # 8. create the random search object
    rf = RandomForestClassifier(random_state=42)

    rfRandomSearch = RandomizedSearchCV(
        estimator=rf,
        param_distributions=paramGrid,
        n_iter=20,               # Try 20 random combinations
        cv=3,                    # 3-fold cross-validation
        random_state=42,
        n_jobs=-1                # use all CPU cores
    )

    # 9. run the search
    rfRandomSearch.fit(XTrain, yTrain)

    print("\nBest Parameters Found:")
    print(rfRandomSearch.best_params_)

    # 10. evaluate the tuned model
    bestRf = rfRandomSearch.best_estimator_
    predictions = bestRf.predict(XTest)

    print("\n=== Tuned Random Forest ===")
    print("precision:", precision_score(yTest, predictions))
    print("recall:", recall_score(yTest, predictions))
    print("f1Score:", f1_score(yTest, predictions))
    print("rocAuc:", roc_auc_score(yTest, predictions))

if __name__ == "__main__":
    tuneRandomForest()
    