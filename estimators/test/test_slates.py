import os, sys, random, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimators.slates import pseudo_inverse
from estimators.slates import gaussian
from estimators.bandits import ips
from estimators.test.utils import Helper

def test_single_slot_pi_equivalent_to_ips():
    ''' PI should be equivalent to IPS when there is only a single slot '''

    pi_estimator = pseudo_inverse.Estimator()
    ips_estimator = ips.Estimator()

    p_logs = [0.8, 0.25, 0.5, 0.2]
    p_preds = [0.6, 0.4, 0.3, 0.9]
    rewards = [0.1, 0.2, 0, 1.0]

    for p_log, r, p_pred in zip(p_logs, rewards, p_preds):
        pi_estimator.add_example([p_log], r, [p_pred])
        ips_estimator.add_example(p_log, r, p_pred)
        Helper.assert_is_close(pi_estimator.get() , ips_estimator.get())

def test_slates():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''

    # The tuple (Estimator, expected value) for each estimator is stored in listofestimators
    listofestimators = [(pseudo_inverse.Estimator(), 1)]

    def datagen(num_slots):
        # num_slots represents the len(p_logs) or len(p_pred) for each example
        data = {'p_log': [], 'r': 0.0, 'p_pred': []}
        for s in range(num_slots):
            data['p_log'].append(1)
            data['p_pred'].append(1)
        data['r'] = 1
        return  data

    # 4 examples; each example of the type->
    # p_logs = [1,1,1,1]
    # p_pred = [1,1,1,1]
    # reward = 1
    estimates = Helper.get_estimate(lambda: datagen(num_slots=4), listofestimators=[l[0] for l in listofestimators], num_examples=4)

    for Estimator, estimate in zip(listofestimators, estimates):
        Helper.assert_is_close(Estimator[1], estimate)

def test_narrowing_intervals():
    ''' To test for narrowing intervals; Number of examples increase => narrowing CI '''

    listofintervals = [gaussian.Interval()]

    def datagen(num_slots, epsilon, delta=0.5):

        data = {'p_log': [], 'r': 0.0, 'p_pred': []}

        for s in range(num_slots):
            # Logged Policy for each slot s
            # 0 - (1-epsilon) : Reward is Bernoulli(delta)
            # 1 - epsilon : Reward is Bernoulli(1-delta)

            # p_pred: 1 if action is chosen, 0 if action not chosen

            # policy to estimate
            # (delta), (1-delta) reward from a Bernoulli distribution - for probability p_pred; looking at the matches per slot s

            chosen = int(random.random() < epsilon)
            data['p_log'].append(epsilon if chosen == 1 else 1 - epsilon)
            data['r'] += int(random.random() < 1-delta) if chosen == 1 else int(random.random() < delta)
            data['p_pred'].append(int(chosen==1))

        return data

    intervals_n1 = Helper.get_estimate(lambda: datagen(num_slots=4, epsilon=0.5), listofintervals, num_examples=100)
    intervals_n2 = Helper.get_estimate(lambda: datagen(num_slots=4, epsilon=0.5), listofintervals, num_examples=10000)

    for interval_n1, interval_n2 in zip(intervals_n1, intervals_n2):
        width_n1 = interval_n1[1] - interval_n1[0]
        width_n2 = interval_n2[1] - interval_n2[0]
        assert width_n1 > 0
        assert width_n2 > 0
        assert width_n2 < width_n1
