# Explainable Credit Risk Assessment

An end-to-end **Machine Learning project for Credit Risk Assessment** that predicts whether a loan applicant is likely to default. The project combines data validation, exploratory data analysis, machine learning, hyperparameter tuning, probability calibration, threshold optimization, and SHAP-based model explainability.

The trained model is deployed through a **FastAPI REST API**, allowing users to submit loan applicant information and receive a default probability and risk classification.

## 🚀 Project Overview

The objective of this project is to build an explainable credit risk model that can identify applicants who may have a higher probability of loan default.

The project covers:

* Exploratory Data Analysis (EDA)
* Data validation and cleaning
* Missing-value handling
* Outlier detection
* Class imbalance handling
* Feature preprocessing
* Logistic Regression baseline model
* XGBoost classification
* Stratified 5-fold cross-validation
* Hyperparameter tuning using RandomizedSearchCV
* Classification threshold optimization
* Probability calibration
* SHAP global and local explainability
* False Positive and False Negative analysis
* Model serialization using Joblib
* FastAPI deployment

## 📊 Dataset

The dataset contains information about loan applicants, including:

* Age
* Annual income
* Home ownership
* Employment length
* Loan intent
* Loan grade
* Loan amount
* Loan interest rate
* Loan percentage of income
* Previous default history
* Credit history length
* Loan status / default indicator

The target variable is:

```text
loan_status
```

where the model predicts whether an applicant belongs to the default or non-default class.

## 🧹 Data Validation

Before model training, the dataset is checked and cleaned by:

* Removing duplicate records
* Removing invalid ages
* Removing unrealistic employment lengths
* Removing invalid loan amounts
* Checking interest-rate ranges
* Checking missing values
* Examining numerical outliers
* Checking class distribution
* Reviewing feature correlations

## 🔧 Machine Learning Pipeline

### Logistic Regression

Logistic Regression is used as the baseline model.

For numerical features, the pipeline performs:

* Median imputation
* Standard scaling

For categorical features:

* Missing-value imputation
* One-hot encoding

### XGBoost

XGBoost is used as the tree-based classification model.

The preprocessing pipeline performs:

* Median imputation for numerical features
* Missing-value handling for categorical features
* One-hot encoding of categorical variables

Unlike Logistic Regression, numerical features are not scaled for XGBoost.

## ⚖️ Class Imbalance

Credit default datasets can contain significantly fewer default cases than non-default cases.

To address this, class weighting is incorporated during model training so that the model gives additional importance to the minority class.

## 🔍 Hyperparameter Tuning

XGBoost hyperparameters are optimized using:

```text
RandomizedSearchCV
```

The tuning process searches parameters such as:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`

Five-fold stratified cross-validation is used during model evaluation and tuning.

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Precision-Recall Curve

The project places particular attention on precision and recall because identifying default-risk applicants is an imbalanced classification problem.

## 🎯 Classification Threshold Optimization

Instead of relying only on the default classification threshold, different probability thresholds are evaluated.

The threshold producing the highest F1 score is selected and saved as:

```text
best_threshold.pkl
```

This allows the deployed application to use the selected classification threshold.

## 📏 Probability Calibration

The XGBoost model is calibrated using:

```text
CalibratedClassifierCV
```

with sigmoid calibration.

Calibration is used to improve the reliability of predicted probabilities.

The final trained model is saved as:

```text
credit_risk_model.pkl
```

## 🧠 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to understand model predictions.

### Global Explainability

SHAP summary plots help identify which features have the greatest overall influence on the model.

### Local Explainability

A SHAP waterfall plot is used to explain the prediction for an individual applicant and show how individual features contribute to the prediction.

## ❌ False Positive & False Negative Analysis

The project also analyzes incorrect predictions:

### False Positive

The model predicts an applicant as high risk, but the actual class is non-default.

### False Negative

The model predicts an applicant as low risk, but the actual class is default.

Analyzing these errors helps identify situations where the model may struggle.

## 🌐 FastAPI Application

The trained model is exposed through a FastAPI application.

The API accepts applicant information including:

```text
person_age
person_income
person_home_ownership
person_emp_length
loan_intent
loan_grade
loan_amnt
loan_int_rate
loan_percent_income
cb_person_default_on_file
cb_person_cred_hist_length
```

The `/predict` endpoint returns:

```json
{
  "default_probability": 0.XX,
  "default_prediction": 0,
  "threshold": 0.XX,
  "Result": "Low Risk"
}
```

The API implementation loads both the trained model and classification threshold when the application starts.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* FastAPI
* Pydantic
* Uvicorn
* Joblib

The deployment requirements include FastAPI, Uvicorn, Pydantic, Pandas, Scikit-learn, XGBoost, and Joblib.

## 📁 Project Structure

```text
Credit-Risk-Assessment/
│
├── Credit_Risk.ipynb
├── credit_risk_dataset.csv
├── credit_risk_model.pkl
├── best_threshold.pkl
├── main.py
├── requirements.txt
├── runtime.txt
├── render.yaml
├── .gitignore
└── README.md
```

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/credit-risk-assessment.git
cd credit-risk-assessment
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI application

```bash
uvicorn main:app --reload
```

The application can then be accessed through the local server.

## ☁️ Deployment

The project is configured for deployment using **Render**.

The Render configuration uses Python, installs dependencies from `requirements.txt`, and starts the application with Uvicorn.

The project uses Python 3.11.9 for the runtime environment.

## 🔮 Future Improvements

Possible future improvements include:

* Adding a dedicated frontend dashboard
* Adding authentication for the API
* Improving model monitoring
* Adding automated model retraining
* Adding more detailed applicant-level explanations
* Adding model performance monitoring after deployment
* Adding automated data validation
* Improving API documentation and testing

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**. Predictions should not be used as the sole basis for real-world lending or financial decisions.

## 👨‍💻 Author

**Shubham Singh**

B.Tech – Information Technology

Interested in **Data Analytics, Data Science, Machine Learning, Python, SQL, and Data Engineering**.
