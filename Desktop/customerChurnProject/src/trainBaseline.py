import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import preprocess as prep  # this uses your loadAndCleanData function

def buildAndTrainBaselineModel():
    # 1. load and clean the raw churn data
    df = prep.loadAndCleanData()

    # 2. drop the customerID column (not useful for prediction)
    df = df.drop(columns=["customerID"])

    # 3. split into features (X) and target label (y)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 4. one-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # 5. create train and test sets
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. create a baseline logistic regression model
    model = LogisticRegression(max_iter=300)

    # 7. train the model (learn patterns from the training data)
    model.fit(XTrain, yTrain)

    # 8. make predictions on the unseen test data
    predictions = model.predict(XTest)

    # 9. compute evaluation metrics
    precision = precision_score(yTest, predictions)
    recall = recall_score(yTest, predictions)
    f1 = f1_score(yTest, predictions)
    rocAuc = roc_auc_score(yTest, predictions)

    print("precisionScore:", precision)
    print("recallScore:", recall)
    print("f1Score:", f1)
    print("rocAucScore:", rocAuc)

if __name__ == "__main__":
    buildAndTrainBaselineModel()
