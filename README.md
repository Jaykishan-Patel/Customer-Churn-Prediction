📊 Customer Churn Prediction using Machine Learning
📌 Project Overview

Customer churn is a major challenge in the telecom industry, as acquiring new customers is significantly more expensive than retaining existing ones.
This project aims to predict whether a customer will churn or not using historical customer data, enabling businesses to take proactive retention actions.

The solution is built as an end-to-end, leakage-free machine learning pipeline, focusing on Recall and F1-score to align with real business objectives.

🎯 Objectives

Predict customer churn (Yes / No)

Identify high-risk customers in advance

Reduce revenue loss through data-driven decision making

Build a production-ready ML pipeline

🧠 Problem Statement

Given customer demographic information, service usage details, and billing data, the task is to build a classification model that accurately predicts customer churn.
Since churn data is imbalanced, accuracy alone is not sufficient; therefore, Recall and F1-score are prioritized.

📂 Dataset

Source: Kaggle (Telco Customer Churn Dataset)

Size: 7,043 rows × 21 columns

Target Variable: Churn

1 → Customer churned

0 → Customer retained

🛠️ Tech Stack

Programming: Python

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn

Visualization: Matplotlib, Seaborn

Model Saving: joblib

🔍 Key Steps in the Project
1️⃣ Data Preprocessing

Removed duplicate and irrelevant columns

Handled missing values and corrected data types

Converted TotalCharges to numeric

Binned tenure into categorical groups

Applied One-Hot Encoding for categorical variables

Scaled numerical features where required

2️⃣ Exploratory Data Analysis (EDA)

Analyzed numerical and categorical features

Identified churn patterns based on tenure, contract type, and monthly charges

Observed strong class imbalance (≈ 73:27)

3️⃣ Handling Imbalanced Data

Applied SMOTE / SMOTEENN to improve minority class (churn) prediction

Compared model performance before and after resampling

4️⃣ Model Building

Trained and evaluated multiple models:

Logistic Regression

Support Vector Machine (SVM)

Decision Tree

Random Forest

Gradient Boosting

XGBoost

Used Pipeline and ColumnTransformer to prevent data leakage.

5️⃣ Model Evaluation

Cross-validation for stable performance

Metrics used:

Precision

Recall

F1-score

Confusion Matrix

Recall and F1-score prioritized due to business importance

6️⃣ Final Model Selection

Random Forest selected as the final model

Provided the best balance of Recall and F1-score

Evaluated on test data using confusion matrix and classification report

📈 Model Performance (Final Model)
Metric	Value
Accuracy	0.72
Precision (Churn)	0.48
Recall (Churn)	0.82
F1-score (Churn)	0.61

✔ High recall ensures most churn-prone customers are identified.

💾 Model Deployment

Final pipeline saved using joblib

Supports prediction on new/unseen customer data

Ensures consistent preprocessing during inference

joblib.dump(final_rf_pipeline, "churn_random_forest_pipeline.pkl")

🔮 Predicting for New Customers

Load the saved pipeline

Provide customer details in the same format as training data

Pipeline automatically handles preprocessing and prediction

📌 What I Learned

How to solve a real-world imbalanced classification problem

Importance of Recall and F1-score over accuracy

Preventing data leakage using pipelines

Handling class imbalance with SMOTE/SMOTEENN

Model comparison and business-driven model selection

Saving and deploying ML models for real-world use

🚀 Future Improvements

Hyperparameter tuning using GridSearchCV

Threshold optimization for better precision–recall tradeoff

Deploy model using Flask / FastAPI

Monitor model performance and data drift

👤 Author

Jaykishan Patel
Aspiring Data Scientist / ML Engineer
