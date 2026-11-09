"""Benchmark models with stratified k-fold CV, calibrate, score the retention
strategy on out-of-fold predictions, and save the final model.

    python -m src.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import FEATURES, TARGET, load_clean
from .model import SEED, calibrated_random_forest, logistic_regression, random_forest
from .value import compare_rankings

ROOT = Path(__file__).resolve().parents[1]
N_FOLDS = 5


def oof_probabilities(model, X, y, cv):
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def fold_aucs(model, X, y, cv):
    aucs = []
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.array(aucs)


def main():
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print(f"Records: {len(df)} | churn base rate: {y.mean():.1%}")

    benchmark = {}
    for name, model in [("logistic_regression", logistic_regression()),
                        ("random_forest", random_forest())]:
        aucs = fold_aucs(model, X, y, cv)
        benchmark[name] = {"roc_auc_mean": round(aucs.mean(), 4),
                           "roc_auc_std": round(aucs.std(), 4)}
        print(f"{name:20s} ROC-AUC {aucs.mean():.4f} ± {aucs.std():.4f}")

    # Raw RF scores vs isotonic-calibrated probabilities, both out-of-fold.
    p_raw = oof_probabilities(random_forest(), X, y, cv)
    p_cal = oof_probabilities(calibrated_random_forest(), X, y, cv)
    calibration = {
        "brier_raw_rf": round(brier_score_loss(y, p_raw), 4),
        "brier_isotonic": round(brier_score_loss(y, p_cal), 4),
        "roc_auc_isotonic_oof": round(roc_auc_score(y, p_cal), 4),
    }
    print(f"Brier raw RF {calibration['brier_raw_rf']:.4f} -> "
          f"isotonic {calibration['brier_isotonic']:.4f}")

    retention = compare_rankings(p_cal, y.to_numpy(), df["MonthlyCharges"].to_numpy())
    print(f"Contacting top {retention['target_share']:.0%}: "
          f"probability ranking ${retention['net_saved_probability_ranking']:,.0f}, "
          f"expected-value ranking ${retention['net_saved_expected_value_ranking']:,.0f} "
          f"({retention['uplift_pct']:+.1f}%)")

    # The uplift depends on the campaign budget, so report it across budgets too.
    sensitivity = [compare_rankings(p_cal, y.to_numpy(), df["MonthlyCharges"].to_numpy(), share)
                   for share in (0.10, 0.20, 0.30, 0.40)]
    for r in sensitivity:
        print(f"  budget {r['target_share']:.0%}: uplift {r['uplift_pct']:+.1f}%")

    metrics = {
        "n_records": int(len(df)),
        "churn_base_rate": round(float(y.mean()), 4),
        "cv_folds": N_FOLDS,
        "benchmark": benchmark,
        "calibration": calibration,
        "retention": retention,
        "retention_sensitivity": sensitivity,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    final = calibrated_random_forest().fit(X, y)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models" / "churn_model.joblib")
    print("Saved models/churn_model.joblib and reports/metrics.json")


if __name__ == "__main__":
    main()
