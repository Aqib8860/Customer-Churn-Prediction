import logging

import pandas as pd

from fastapi import FastAPI, HTTPException

from api.config import CHURN_THRESHOLD
from api.model import predict_churn_probability
from api.schemas import (
    ChurnPrediction,
    CustomerData,
)


# ---------------------------------------------------------
# Configure application logging.
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(
    __name__
)


# ---------------------------------------------------------
# Create FastAPI application.
# ---------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


# ---------------------------------------------------------
# Health check endpoint.
# ---------------------------------------------------------
@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "model": "customer_churn_pipeline",
        "threshold": CHURN_THRESHOLD,
    }


# ---------------------------------------------------------
# Customer churn prediction endpoint.
# ---------------------------------------------------------
@app.post(
    "/predict",
    response_model=ChurnPrediction,
)
def predict_churn(
    customer: CustomerData,
):

    try:

        # -------------------------------------------------
        # Convert validated request into DataFrame.
        #
        # Column names match the training dataset.
        # -------------------------------------------------
        input_data = pd.DataFrame(
            [
                customer.model_dump()
            ]
        )

        # -------------------------------------------------
        # Generate churn probability using the saved
        # ML pipeline.
        # -------------------------------------------------
        churn_probability = (
            predict_churn_probability(
                input_data
            )
        )

        # -------------------------------------------------
        # Convert probability into final business decision.
        #
        # 0.40 is loaded from configuration.
        # -------------------------------------------------
        prediction = int(
            churn_probability
            >= CHURN_THRESHOLD
        )

        # -------------------------------------------------
        # Log prediction for debugging/monitoring.
        # -------------------------------------------------
        logger.info(
            "Churn prediction generated: "
            "prediction=%s probability=%.4f",
            prediction,
            churn_probability,
        )

        # -------------------------------------------------
        # Return API response.
        # -------------------------------------------------
        return {
            "churn": prediction,
            "churn_probability": round(
                churn_probability,
                4,
            ),
        }

    except Exception as exc:

        # -------------------------------------------------
        # Log unexpected server-side errors.
        # -------------------------------------------------
        logger.exception(
            "Churn prediction failed: %s",
            exc,
        )

        # -------------------------------------------------
        # Do not expose internal error details to the client.
        # -------------------------------------------------
        raise HTTPException(
            status_code=500,
            detail="Prediction service failed.",
        )