import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import preprocess as prep
import numpy as np

def tuneGradientBoosting():
    # 1. load and clean data
    df = prep.loadAndCleanData()
    df = df.drop(columns=["customerID"])

    # 2. separate X and y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. encode
    X = pd.get_dummies(X, drop_first=True)

    # 4. split
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. SMOTE
    sm = SMOTE(random_state=42)
    XTrain, yTrain = sm.fit_resample(XTrain, yTrain)

    # 6. scale numeric features
    numericCols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    scaler = StandardScaler()
    XTrain[numericCols] = scaler.fit_transform(XTrain[numericCols])
    XTest[numericCols] = scaler.transform(XTest[numericCols])

    # 7. full hyperparameter search space
    paramGrid = {
        "n_estimators": [100, 150, 200, 250, 300, 350, 400],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        "max_depth": [2, 3, 4, 5, 6],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 6],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    # 8. create model
    gb = GradientBoostingClassifier(random_state=42)

    # 9. random search
    gbSearch = RandomizedSearchCV(
        estimator=gb,
        param_distributions=paramGrid,
        n_iter=25,         # FULL tuning = 25 combinations
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    gbSearch.fit(XTrain, yTrain)

    print("\nBest Gradient Boosting Parameters:")
    print(gbSearch.best_params_)

    # 10. evaluate best model
    bestGb = gbSearch.best_estimator_
    preds = bestGb.predict(XTest)

    print("\n=== Tuned Gradient Boosting ===")
    print("precision:", precision_score(yTest, preds))
    print("recall:", recall_score(yTest, preds))
    print("f1Score:", f1_score(yTest, preds))
    print("rocAuc:", roc_auc_score(yTest, preds))

if __name__ == "__main__":
    tuneGradientBoosting()
