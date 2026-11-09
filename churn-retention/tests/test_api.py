import pytest
from fastapi.testclient import TestClient

from src.api import MODEL_PATH, app

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")
client = TestClient(app)

RISKY = {"tenure": 2, "MonthlyCharges": 95.0, "TotalCharges": 190.0}
LOYAL = {"tenure": 70, "MonthlyCharges": 25.0, "TotalCharges": 1750.0,
         "Contract": "Two year", "InternetService": "No",
         "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
         "DeviceProtection": "No internet service", "TechSupport": "No internet service",
         "StreamingTV": "No internet service", "StreamingMovies": "No internet service",
         "PaymentMethod": "Credit card (automatic)"}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_valid_probability():
    r = client.post("/predict", json=RISKY)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0


def test_risky_customer_scores_above_loyal_one():
    risky = client.post("/predict", json=RISKY).json()
    loyal = client.post("/predict", json=LOYAL).json()
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["send_offer"] and not loyal["send_offer"]


def test_batch_is_sorted_by_expected_value():
    out = client.post("/predict/batch", json=[LOYAL, RISKY]).json()
    evs = [row["expected_value_of_offer"] for row in out]
    assert evs == sorted(evs, reverse=True)


def test_rejects_invalid_input():
    assert client.post("/predict", json={**RISKY, "Contract": "Weekly"}).status_code == 422
