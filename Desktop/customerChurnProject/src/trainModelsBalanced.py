import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import preprocess as prep

def runBalancedModels():
    df = prep.loadAndCleanData()
    df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    numericCols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    scaler = StandardScaler()

    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    XTrain[numericCols] = scaler.fit_transform(XTrain[numericCols])
    XTest[numericCols] = scaler.transform(XTest[numericCols])

    # 1. Logistic Regression with class_weight
    logModel = LogisticRegression(max_iter=300, class_weight="balanced")
    logModel.fit(XTrain, yTrain)
    logPreds = logModel.predict(XTest)

    print("\n=== Logistic Regression (Balanced) ===")
    print("precision:", precision_score(yTest, logPreds))
    print("recall:", recall_score(yTest, logPreds))
    print("f1Score:", f1_score(yTest, logPreds))
    print("rocAuc:", roc_auc_score(yTest, logPreds))

    # 2. Random Forest with class_weight
    rfModel = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    )
    rfModel.fit(XTrain, yTrain)
    rfPreds = rfModel.predict(XTest)

    print("\n=== Random Forest (Balanced) ===")
    print("precision:", precision_score(yTest, rfPreds))
    print("recall:", recall_score(yTest, rfPreds))
    print("f1Score:", f1_score(yTest, rfPreds))
    print("rocAuc:", roc_auc_score(yTest, rfPreds))

    # 3. Gradient Boosting (class_weight not supported directly)
    # We will simulate balancing by giving more weight to the positive class
    gbModel = GradientBoostingClassifier(random_state=42)
    gbModel.fit(XTrain, yTrain)
    gbPreds = gbModel.predict(XTest)

    print("\n=== Gradient Boosting (No class_weight) ===")
    print("precision:", precision_score(yTest, gbPreds))
    print("recall:", recall_score(yTest, gbPreds))
    print("f1Score:", f1_score(yTest, gbPreds))
    print("rocAuc:", roc_auc_score(yTest, gbPreds))

if __name__ == "__main__":
    runBalancedModels()
