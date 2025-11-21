import pandas as pd
import numpy as np

def loadAndCleanData():
    # 1. load dataset
    df = pd.read_csv("data/churn.csv")

    # 2. strip whitespace in all string columns
    #    sometimes values like " Yes" or "No " break encoding
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # 3. fix totalCharges: convert from string to numeric
    #    errors='coerce' turns invalid values (like " ") into NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 4. drop rows where totalCharges is NaN
    df = df.dropna(subset=["TotalCharges"])

    # 5. convert churn column from Yes/No to 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df

if __name__ == "__main__":
    cleanedDf = loadAndCleanData()
    print("cleanedShape:", cleanedDf.shape)
    print("remainingMissing:", cleanedDf.isnull().sum().sum())
    print("exampleRows:\n", cleanedDf.head())
