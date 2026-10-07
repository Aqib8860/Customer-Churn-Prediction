from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_check():
    """
    Verify that the API health endpoint is working.
    """

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json()["status"] == "ok"


def test_churn_prediction():
    """
    Verify that the prediction endpoint accepts
    valid customer data and returns the expected
    response structure.
    """

    payload = {
        "age": 29,
        "tenure_months": 6,
        "monthly_charges": 4200,
        "support_tickets": 5,
        "contract_type": "Monthly",
        "payment_method": "UPI",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["churn"] in [0, 1]

    assert (
        0
        <= result["churn_probability"]
        <= 1
    )


def test_invalid_customer_age():
    """
    Verify that Pydantic validation rejects
    an invalid customer age.
    """

    payload = {
        "age": 10,
        "tenure_months": 6,
        "monthly_charges": 4200,
        "support_tickets": 5,
        "contract_type": "Monthly",
        "payment_method": "UPI",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

