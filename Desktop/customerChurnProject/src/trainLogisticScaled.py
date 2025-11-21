import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import preprocess as prep

def runLogisticScaled():
    # 1. load and clean data
    df = prep.loadAndCleanData()
    df = df.drop(columns=["customerID"])

    # 2. separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. one-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # 4. identify numeric columns (we scale only these)
    numericCols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

    scaler = StandardScaler()

    # 5. split BEFORE scaling (correct way)
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. fit scaler on training numeric columns only
    XTrain[numericCols] = scaler.fit_transform(XTrain[numericCols])

    # 7. transform the test numeric columns (using SAME scaler)
    XTest[numericCols] = scaler.transform(XTest[numericCols])

    # 8. train logistic regression
    model = LogisticRegression(max_iter=300)
    model.fit(XTrain, yTrain)

    # 9. make predictions
    preds = model.predict(XTest)

    # 10. compute metrics
    precision = precision_score(yTest, preds)
    recall = recall_score(yTest, preds)
    f1 = f1_score(yTest, preds)
    rocAuc = roc_auc_score(yTest, preds)

    print("precision:", precision)
    print("recall:", recall)
    print("f1Score:", f1)
    print("rocAuc:", rocAuc)

if __name__ == "__main__":
    runLogisticScaled()
