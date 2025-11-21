import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import preprocess as prep

def runAdvancedModels():
    # 1. load and clean raw data
    df = prep.loadAndCleanData()
    df = df.drop(columns=["customerID"])

    # 2. separate features and labels
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. one-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # 4. split into train and test sets
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. random forest model
    rfModel = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    rfModel.fit(XTrain, yTrain)
    rfPreds = rfModel.predict(XTest)

    rfPrecision = precision_score(yTest, rfPreds)
    rfRecall = recall_score(yTest, rfPreds)
    rfF1 = f1_score(yTest, rfPreds)
    rfAuc = roc_auc_score(yTest, rfPreds)

    print("\n=== Random Forest ===")
    print("precision:", rfPrecision)
    print("recall:", rfRecall)
    print("f1Score:", rfF1)
    print("rocAuc:", rfAuc)

    # 6. gradient boosting model
    gbModel = GradientBoostingClassifier(random_state=42)
    gbModel.fit(XTrain, yTrain)
    gbPreds = gbModel.predict(XTest)

    gbPrecision = precision_score(yTest, gbPreds)
    gbRecall = recall_score(yTest, gbPreds)
    gbF1 = f1_score(yTest, gbPreds)
    gbAuc = roc_auc_score(yTest, gbPreds)

    print("\n=== Gradient Boosting ===")
    print("precision:", gbPrecision)
    print("recall:", gbRecall)
    print("f1Score:", gbF1)
    print("rocAuc:", gbAuc)

if __name__ == "__main__":
    runAdvancedModels()
