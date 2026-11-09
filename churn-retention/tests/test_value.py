import numpy as np

from src.value import cltv, expected_value, revenue_saved


def test_cltv_scales_with_monthly_charges():
    v = cltv([10.0, 20.0])
    assert np.isclose(v[1], 2 * v[0])


def test_expected_value_prefers_valuable_customer_at_same_risk():
    ev = expected_value([0.5, 0.5], cltv([20.0, 100.0]))
    assert ev[1] > ev[0]


def test_revenue_saved_counts_only_real_churners():
    value = np.array([1000.0, 1000.0])
    assert revenue_saved([1.0, 0.0], [0, 1], value, share=0.5) < \
           revenue_saved([0.0, 1.0], [0, 1], value, share=0.5)
