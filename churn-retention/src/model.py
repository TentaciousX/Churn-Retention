"""Model definitions shared by training and serving."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC

SEED = 42


def preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def logistic_regression() -> Pipeline:
    return Pipeline([
        ("prep", preprocessor(scale=True)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def random_forest() -> Pipeline:
    return Pipeline([
        ("prep", preprocessor(scale=False)),
        ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        )),
    ])


def calibrated_random_forest() -> CalibratedClassifierCV:
    """Random Forest whose scores are mapped to probabilities by isotonic regression."""
    return CalibratedClassifierCV(random_forest(), method="isotonic", cv=3)
