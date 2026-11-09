"""Expected-value retention targeting.

A retention campaign can only contact a fixed share of customers. Ranking by
churn probability alone spends that budget on likely churners even when they
pay very little. Ranking by expected value weighs each customer's churn risk
by how much revenue is actually at stake.
"""
import numpy as np
import pandas as pd

# Campaign assumptions. These are inputs, not results: change them and rerun.
HORIZON_MONTHS = 24          # revenue window a saved customer is worth
MONTHLY_DISCOUNT = 0.01      # ~12.7% annual discount rate
SAVE_RATE = 0.30             # share of contacted churners the offer retains
OFFER_COST = 50.0            # cost of one retention offer, in $
TARGET_SHARE = 0.20          # campaign budget: contact 20% of customers


def cltv(monthly_charges) -> np.ndarray:
    """Discounted revenue over the horizon if the customer stays."""
    t = np.arange(1, HORIZON_MONTHS + 1)
    annuity = np.sum(1.0 / (1.0 + MONTHLY_DISCOUNT) ** t)
    return np.asarray(monthly_charges, dtype=float) * annuity


def expected_value(p_churn, customer_value) -> np.ndarray:
    """Expected net gain from sending this customer an offer."""
    return np.asarray(p_churn) * SAVE_RATE * np.asarray(customer_value) - OFFER_COST


def revenue_saved(scores, churned, customer_value, share=TARGET_SHARE) -> float:
    """Net revenue retained when the top `share` of customers by score are contacted.

    Uses the real churn labels, so it measures what each ranking would have earned.
    """
    scores = np.asarray(scores)
    k = int(round(share * len(scores)))
    top = np.argsort(-scores, kind="stable")[:k]
    gross = SAVE_RATE * np.sum(np.asarray(churned)[top] * np.asarray(customer_value)[top])
    return float(gross - OFFER_COST * k)


def compare_rankings(p_churn, churned, monthly_charges, share=TARGET_SHARE) -> dict:
    value = cltv(monthly_charges)
    by_prob = revenue_saved(p_churn, churned, value, share)
    by_ev = revenue_saved(expected_value(p_churn, value), churned, value, share)
    return {
        "target_share": share,
        "net_saved_probability_ranking": round(by_prob, 2),
        "net_saved_expected_value_ranking": round(by_ev, 2),
        "uplift_pct": round(100 * (by_ev - by_prob) / by_prob, 2),
    }
