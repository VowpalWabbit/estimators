import os, sys, random, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimators.slates import pseudo_inverse
from estimators.slates import gaussian
from estimators.bandits import ips
from estimators.utils.helper_tests import SlatesHelper

def test_single_slot_pi_equivalent_to_ips():
    """PI should be equivalent to IPS when there is only a single slot"""
    pi_estimator = pseudo_inverse.Estimator()
    ips_estimator = ips.Estimator()
    is_close = lambda a, b: abs(a - b) <= 1e-6 * (1 + abs(a) + abs(b))

    p_logs = [0.8, 0.25, 0.5, 0.2]
    p_preds = [0.6, 0.4, 0.3, 0.9]
    rewards = [0.1, 0.2, 0, 1.0]

    for p_log, r, p_pred in zip(p_logs, rewards, p_preds):
        pi_estimator.add_example([p_log], r, [p_pred])
        ips_estimator.add_example(p_log, r, p_pred)
        assert is_close(pi_estimator.get() , ips_estimator.get())

def test_slates():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''
    
    # The tuple (Estimator, expected value) for each estimator is stored in listofestimators
    listofestimators = [(pseudo_inverse.Estimator(), 1)]

    # 4 examples; each example of the type->
    # p_logs = [1,1,1,1]
    # p_pred = [1,1,1,1]
    # reward = 1
    SlatesHelper.run_estimator(SlatesHelper.example_generator1, listofestimators, num_examples=4, num_slots=4)
