import os, sys, random, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimators.slates import pseudo_inverse
from estimators.slates import gaussian
from estimators.bandits import ips
from estimators.utils.helper_tests import Helper

helper = Helper()

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

    def example_generator(num_slots):
        # num_slots represents the len(p_logs) or len(p_pred) for each example
        data = {'p_logs': [], 'r': 0.0, 'p_preds': []}
        for s in range(num_slots):
            data['p_logs'].append(1)
            data['p_preds'].append(1)
        data['r'] = 1
        return  data

    # 4 examples; each example of the type->
    # p_logs = [1,1,1,1]
    # p_pred = [1,1,1,1]
    # reward = 1
    estimates = helper.get_estimate(datagen=lambda: example_generator(num_slots=4), listofestimators=listofestimators, num_examples=4)

    is_close = lambda a, b: abs(a - b) <= 1e-6 * (1 + abs(a) + abs(b))
    for Estimator, estimate in zip(listofestimators, estimates):
        assert is_close(Estimator[1], estimate)

def test_intervals():
    ''' To test for narrowing intervals '''

    listofintervals = [gaussian.Interval()]

    def example_generator(num_slots, epsilon, delta=0.5):

        data = {'p_logs': [], 'r': 0.0, 'p_preds': []}

        for s in range(num_slots):
            # Logged Policy for each slot s
            # 0 - (1-epsilon) : Reward is Bernoulli(delta)
            # 1 - epsilon : Reward is Bernoulli(1-delta)

            # p_pred: 1 if action is chosen, 0 if action not chosen

            # policy to estimate
            # (delta), (1-delta) reward from a Bernoulli distribution - for probability p_pred; looking at the matches per slot s

            chosen = int(random.random() < epsilon)
            data['p_logs'].append(epsilon if chosen == 1 else 1 - epsilon)
            data['r'] += int(random.random() < 1-delta) if chosen == 1 else int(random.random() < delta)
            data['p_preds'].append(int(chosen==1))

        return data

    widths_n1, widths_n2 = helper.calc_CI_width(lambda: example_generator(num_slots=4, epsilon=0.5), listofintervals, n1=100, n2=10000)
    for width_n1, width_n2 in zip(widths_n1, widths_n2):
        assert width_n1 > 0
        assert width_n2 > 0
        assert width_n2 < width_n1
