from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Customer prediction request.
# ---------------------------------------------------------
class CustomerData(BaseModel):

    age: float = Field(..., ge=18, le=100)
    tenure_months: int = Field(..., ge=1)
    monthly_charges: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    contract_type: str
    payment_method: str

# ---------------------------------------------------------
# Prediction response.
# ---------------------------------------------------------
class ChurnPrediction(BaseModel):

    churn: int
    churn_probability: float

