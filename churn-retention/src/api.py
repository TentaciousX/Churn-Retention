"""FastAPI service: churn probability, customer value and a retention decision.

    uvicorn src.api:app --reload
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data import FEATURES
from .value import OFFER_COST, cltv, expected_value

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"

app = FastAPI(title="Churn Prediction & Expected-Value Retention", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    tenure: int = Field(ge=0, examples=[5])
    MonthlyCharges: float = Field(gt=0, examples=[89.1])
    TotalCharges: float = Field(ge=0, examples=[445.5])
    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: YesNo = "No"
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    PhoneService: YesNo = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ] = "Electronic check"


class Prediction(BaseModel):
    churn_probability: float
    customer_value: float
    expected_value_of_offer: float
    send_offer: bool


def score(customers: list[Customer]) -> list[Prediction]:
    df = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    p = get_model().predict_proba(df)[:, 1]
    value = cltv(df["MonthlyCharges"])
    ev = expected_value(p, value)
    return [
        Prediction(churn_probability=round(float(pi), 4), customer_value=round(float(vi), 2),
                   expected_value_of_offer=round(float(ei), 2), send_offer=bool(ei > 0))
        for pi, vi, ei in zip(p, value, ev)
    ]


@app.get("/health")
def health():
    return {"status": "ok", "offer_cost": OFFER_COST}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return score([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]):
    """Score many customers, highest expected value first."""
    return sorted(score(customers), key=lambda r: r.expected_value_of_offer, reverse=True)
