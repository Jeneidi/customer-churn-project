import pandas as pd                 # imports the pandas library so we can load and manipulate tabular data
import matplotlib.pyplot as plt     # imports matplotlib so we can create graphs

def runEda():
    # 1. Load the churn dataset from the data folder
    df = pd.read_csv("data/churn.csv") 
    # At this moment, df becomes a DataFrame containing all 7043 rows and 21 columns.

    # 2. Print the shape (number of rows and columns)
    print("dataShape:", df.shape)
    # Expected output: (7043, 21) meaning 7043 rows, 21 columns.

    # 3. Print all column names so we know what features exist
    print("\ncolumnNames:", df.columns.tolist())
    # This prints a Python list containing strings for each column name.

    # 4. Print the data type of each column (object, int64, float64, etc.)
    print("\ndataTypes:\n", df.dtypes)
    # This helps us detect which columns need cleaning (ex: TotalCharges shows as object but should be numeric).

    # 5. Print how many missing values each column has
    print("\nmissingValues:\n", df.isnull().sum())
    # This tells us if any column contains null (NaN) values that need fixing.

    # 6. Count how many customers churned vs did not churn
    churnCounts = df["Churn"].value_counts()
    print("\nchurnCounts:\n", churnCounts)
    # Expected: More "No" than "Yes", meaning the dataset is imbalanced.

    # 7. Create a basic bar chart of churn counts
    plt.figure(figsize=(6,4))       # sets the width and height of the chart
    churnCounts.plot(kind="bar")    # creates a bar chart from the churnCounts data
    plt.title("churnDistribution")  # sets the title of the graph
    plt.xlabel("churn")             # labels x-axis
    plt.ylabel("count")             # labels y-axis
    plt.show()                      # displays the chart in a new window

# This block makes sure runEda() runs only if we run the script directly:
if __name__ == "__main__":
    runEda()
