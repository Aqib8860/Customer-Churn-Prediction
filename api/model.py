from pathlib import Path

import joblib


# ---------------------------------------------------------
# Locate the saved ML pipeline.
#
# This file contains:
# preprocessing + Logistic Regression
# ---------------------------------------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "customer_churn_pipeline.joblib"
)


# ---------------------------------------------------------
# Load the trained pipeline once when the API starts.
# ---------------------------------------------------------
model = joblib.load(
    MODEL_PATH
)


def predict_churn_probability(
    input_data,
) -> float:
    """
    Generate churn probability for one customer.
    """

    # -----------------------------------------------------
    # Class 1 represents churn.
    # -----------------------------------------------------
    probability = model.predict_proba(
        input_data
    )[0][1]

    return float(probability)

