import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(
    "data/customer_churn.csv"
)


# ---------------------------------------------------------
# Separate features and target
#
# X = information about the customer
# y = whether the customer churned
# ---------------------------------------------------------
X = df.drop(
    columns=["churn"]
)

y = df["churn"]


# ---------------------------------------------------------
# Define feature groups
#
# Numerical columns need numerical preprocessing.
# Categorical columns need categorical preprocessing.
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
# Missing numerical values:
#     replace with median.
#
# Scaling:
#     puts numerical features on a comparable scale.
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
# Missing categorical values:
#     replace with most frequent category.
#
# OneHotEncoder:
#     converts categories into numerical columns.
#
# sparse_output=False:
#     keeps the result dense so all three models
#     can use the same preprocessing output.
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
# Combine numerical + categorical preprocessing.
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
# Train/Test split
#
# stratify=y keeps the churn ratio approximately
# equal in both train and test datasets.
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ---------------------------------------------------------
# Define models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
    ),
}


# ---------------------------------------------------------
# Train and evaluate each model.
# ---------------------------------------------------------
for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # -----------------------------------------------------
    # Complete ML pipeline:
    #
    # raw data
    #     ↓
    # preprocessing
    #     ↓
    # model
    # -----------------------------------------------------
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # -----------------------------------------------------
    # Train the complete pipeline.
    # -----------------------------------------------------
    pipeline.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # Generate predictions on unseen test data.
    # -----------------------------------------------------
    predictions = pipeline.predict(
        X_test
    )

    # -----------------------------------------------------
    # Calculate evaluation metrics.
    # -----------------------------------------------------
    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    # -----------------------------------------------------
    # Display metrics.
    # -----------------------------------------------------
    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    # -----------------------------------------------------
    # Detailed classification report.
    # -----------------------------------------------------
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
        )
    )

    # -----------------------------------------------------
    # Confusion matrix
    #
    # [[TN, FP],
    #  [FN, TP]]
    # -----------------------------------------------------
    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )