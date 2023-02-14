from estimators.slates import pseudo_inverse
from estimators.bandits import ips
import pytest


def test_single_slot_pi_equivalent_to_ips():
    """PI should be equivalent to IPS when there is only a single slot"""

    pi_estimator = pseudo_inverse.Estimator()
    ips_estimator = ips.Estimator()

    p_logs = [0.8, 0.25, 0.5, 0.2]
    p_preds = [0.6, 0.4, 0.3, 0.9]
    rewards = [0.1, 0.2, 0, 1.0]

    for p_log, r, p_pred in zip(p_logs, rewards, p_preds):
        pi_estimator.add_example([p_log], r, [p_pred])
        ips_estimator.add_example(p_log, r, p_pred)
        assert pi_estimator.get() == pytest.approx(ips_estimator.get())
