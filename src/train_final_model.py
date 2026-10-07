import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(
    "data/customer_churn.csv"
)


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------
X = df.drop(
    columns=["churn"]
)

y = df["churn"]


# ---------------------------------------------------------
# Feature groups
# ---------------------------------------------------------
numerical_features = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
]

categorical_features = [
    "contract_type",
    "payment_method",
]


# ---------------------------------------------------------
# Numerical preprocessing
#
# Missing values → median
# Scaling → StandardScaler
# ---------------------------------------------------------
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


# ---------------------------------------------------------
# Categorical preprocessing
#
# Missing values → most frequent category
# Categories → One-Hot Encoding
# ---------------------------------------------------------
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# Combine both preprocessing pipelines.
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# Final model pipeline
#
# Everything required for prediction is contained
# inside this single pipeline.
# ---------------------------------------------------------
model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


# ---------------------------------------------------------
# Train/Test split
#
# stratify keeps the churn distribution balanced
# between training and testing data.
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ---------------------------------------------------------
# Train the complete pipeline.
# ---------------------------------------------------------
model_pipeline.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# Save the complete pipeline.
#
# API later isi file ko load karke prediction karegi.
# ---------------------------------------------------------
output_path = (
    "models/customer_churn_pipeline.joblib"
)

joblib.dump(
    model_pipeline,
    output_path,
)


print(
    f"Model saved: {output_path}"
)