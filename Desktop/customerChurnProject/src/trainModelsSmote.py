import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import preprocess as prep

def runModelsWithSmote():
    # 1. load and clean data
    df = prep.loadAndCleanData()
    df = df.drop(columns=["customerID"])

    # 2. separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. one-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # 4. split BEFORE SMOTE
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. apply SMOTE ONLY ON TRAINING DATA
    smote = SMOTE(random_state=42)
    XTrain, yTrain = smote.fit_resample(XTrain, yTrain)

    # 6. scale numeric columns
    numericCols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    scaler = StandardScaler()
    XTrain[numericCols] = scaler.fit_transform(XTrain[numericCols])
    XTest[numericCols] = scaler.transform(XTest[numericCols])

    # --- MODEL 1: Logistic Regression ---
    logModel = LogisticRegression(max_iter=300)
    logModel.fit(XTrain, yTrain)
    logPreds = logModel.predict(XTest)

    print("\n=== Logistic Regression (SMOTE) ===")
    print("precision:", precision_score(yTest, logPreds))
    print("recall:", recall_score(yTest, logPreds))
    print("f1Score:", f1_score(yTest, logPreds))
    print("rocAuc:", roc_auc_score(yTest, logPreds))

    # --- MODEL 2: Random Forest ---
    rfModel = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )
    rfModel.fit(XTrain, yTrain)
    rfPreds = rfModel.predict(XTest)

    print("\n=== Random Forest (SMOTE) ===")
    print("precision:", precision_score(yTest, rfPreds))
    print("recall:", recall_score(yTest, rfPreds))
    print("f1Score:", f1_score(yTest, rfPreds))
    print("rocAuc:", roc_auc_score(yTest, rfPreds))

    # --- MODEL 3: Gradient Boosting ---
    gbModel = GradientBoostingClassifier(random_state=42)
    gbModel.fit(XTrain, yTrain)
    gbPreds = gbModel.predict(XTest)

    print("\n=== Gradient Boosting (SMOTE) ===")
    print("precision:", precision_score(yTest, gbPreds))
    print("recall:", recall_score(yTest, gbPreds))
    print("f1Score:", f1_score(yTest, gbPreds))
    print("rocAuc:", roc_auc_score(yTest, gbPreds))

if __name__ == "__main__":
    runModelsWithSmote()
