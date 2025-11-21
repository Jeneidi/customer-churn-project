Customer Churn Prediction

An end-to-end machine learning project

Overview:
This project predicts customer churn for a telecom company using a full machine learning pipeline I built step by step. I wanted to understand how real ML workflows work, not just train a single model in a notebook. The project includes data cleaning, exploratory analysis, preprocessing, model training, class imbalance handling, tuning, and a final prediction script.

Why I Built This:
I already knew the basics of Python, NumPy, Pandas, and Matplotlib, but I had never built a “real” ML project from start to finish. So, after every new step I made sure to type in exaclty what the functionality of the line is, so I am aware of its uses. I picked churn prediction because it’s a realistic business problem and the dataset isn’t perfectly clean, which forced me to actually think about data issues instead of relying on pre-cleaned examples.

Dataset:
I used the Telco Customer Churn dataset (commonly available online).
Roughly 7,000 rows and 21 columns.
Target variable: Churn (Yes/No).
The data is somewhat messy and imbalanced, which became an important part of the project.

What I Did (Step by Step)
1. Exploratory Data Analysis
-Before touching machine learning, I spent time understanding the dataset:
-Looked at dtypes and missing values
-Checked distribution of churn
-Verified column meanings
-Noticed that TotalCharges was a string column with spaces and needed cleaning

2. Cleaning and Preprocessing
This included:
-Stripping whitespace from all string columns
-Converting TotalCharges to numeric (and dropping rows that couldn’t convert)
-Mapping "Yes"/"No" to 1/0 for the target
-Removing customerID since it’s not useful
-I wrapped the cleaning in a function so I could reuse it everywhere else

3. One-Hot Encoding and Train/Test Split
I applied pd.get_dummies(drop_first=True) to categorical columns and split the dataset 80/20.
After encoding, I ended up with about 30 final features.
At this point, the project started looking like a real pipeline.

4. Baseline Model
I trained a logistic regression model to get a baseline.
The results weren’t great (especially for recall), but it was important because it gave me a reference point for improvement.

5. Scaling and Understanding Why
I scaled numeric columns using StandardScaler.
This helped make logistic regression more stable, but I learned that scaling doesn't automatically fix performance; it just improves optimization for certain models.

6. Trying Better Models
I trained Random Forest and Gradient Boosting models.
Random Forest wasn’t great on this dataset.
Gradient Boosting showed potential but still struggled with class imbalance.

7. Handling Class Imbalance
This was the biggest improvement in the project.
I used SMOTE to oversample the minority (churn) class in the training set.
This significantly increased recall and balanced the model.

8. Hyperparameter Tuning
I tuned Random Forest and Gradient Boosting with RandomizedSearchCV.
Surprisingly, the tuned GBM performed slightly worse than the original GBM using SMOTE.
This taught me that tuning isn’t always automatically better.

9. Final Model
I selected Gradient Boosting + SMOTE as the final model because it gave the best combination of recall, F1 score, and ROC AUC.
I then retrained it cleanly and saved:
-The model
-The scaler
-The feature column list

10. Prediction Script
-I wrote a predict.py file that:
-Loads the saved model
-Loads the saved scaler
-Loads the saved feature schema
-Takes in a new customer (as a Python dictionary)
-One-hot encodes it
-Aligns columns with training
-Scales numeric features
-Outputs a churn prediction and probability
-This is the part that made it feel like an actual ML product.

How to Run:
-Install the necessary packages:
-pip install pandas numpy matplotlib scikit-learn imbalanced-learn joblib
-Place churn.csv inside data/.
-Run the scripts.

What I Learned:
-How to structure a multi-file ML project instead of doing everything in one notebook
-How encoding and scaling actually affect model behavior
-How to handle imbalanced datasets in a real way
-Why it’s important to save model artifacts (model, scaler, columns)
-How to write a clean inference pipeline
-That ML is mostly data prep, not just training models
