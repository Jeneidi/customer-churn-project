import pandas as pd
from sklearn.model_selection import train_test_split

def loadCleanData():
    df = pd.read_csv("data/churn.csv")
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def prepareFeatures(df):
    # 1. drop ID column (it has no predictive value)
    df = df.drop(columns=["customerID"])

    # 2. separate input features (X) and target label (y)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. convert all categorical columns into numeric columns
    X = pd.get_dummies(X, drop_first=True)

    return X, y

def splitData(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = loadCleanData()
    X, y = prepareFeatures(df)
    
    XTrain, XTest, yTrain, yTest = splitData(X, y)

    print("XTrainShape:", XTrain.shape)
    print("XTestShape:", XTest.shape)
    print("yTrainShape:", yTrain.shape)
    print("yTestShape:", yTest.shape)
    print("\nexampleColumns:", XTrain.columns[:10])
    