# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a customer is likely to churn.

The project covers the complete ML lifecycle:

- Data generation
- Data inspection
- Missing-value handling
- Feature preprocessing
- Train/test splitting
- Model comparison
- Model evaluation
- Threshold tuning
- Model persistence
- FastAPI deployment
- API validation
- Logging
- Automated API tests

---

## Project Architecture

```text
Customer Data
      ↓
Data Inspection
      ↓
Train / Test Split
      ↓
Preprocessing Pipeline
      ├── Numerical Imputation
      ├── Standard Scaling
      ├── Categorical Imputation
      └── One-Hot Encoding
      ↓
Model Comparison
      ├── Logistic Regression
      ├── Random Forest
      └── Gradient Boosting
      ↓
Model Evaluation
      ├── Precision
      ├── Recall
      ├── F1 Score
      └── Confusion Matrix
      ↓
Final Logistic Regression Model
      ↓
Joblib Model Persistence
      ↓
FastAPI
      ↓
POST /predict
      ↓
Churn Probability
      ↓
Business Threshold
      ↓
Churn Decision