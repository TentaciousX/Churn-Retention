"""Load and clean the IBM Telco Customer Churn dataset."""
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco.csv"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Churn"


def load_raw() -> pd.DataFrame:
    """Read the CSV from disk, downloading it on first use."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def load_clean() -> pd.DataFrame:
    """Fix types and drop the 11 brand-new customers with blank TotalCharges."""
    df = load_raw()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df
