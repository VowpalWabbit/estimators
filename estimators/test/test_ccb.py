import os, sys, random, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimators.bandits import ips
from estimators.bandits import snips
from estimators.bandits import mle
from estimators.bandits import cressieread
from estimators.bandits import gaussian
from estimators.bandits import clopper_pearson
from estimators.ccb import first_slot
from estimators.test.utils import Helper

def test_ccb_unit_test():
    listofestimators = [(first_slot.Estimator(ips.Estimator()), 2.0),
                        (first_slot.Estimator(snips.Estimator()), 1.0),
                        (first_slot.Estimator(mle.Estimator()), 1.0),
                        (first_slot.Estimator(cressieread.Estimator()), 1.0)]
    
    p_log = [0.3]
    p_pred = [0.6]
    reward = [1]

    for Estimator in listofestimators:
        Estimator[0].add_example(p_log, reward, p_pred)
        assert Estimator[0].get() == Estimator[1]

def test_ccb():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''

    # The tuple (Estimator, expected value) for each estimator is stored in listofestimators
    listofestimators = [(first_slot.Estimator(ips.Estimator()), 1.0),
                        (first_slot.Estimator(snips.Estimator()), 1.0),
                        (first_slot.Estimator(mle.Estimator()), 1.0),
                        (first_slot.Estimator(cressieread.Estimator()), 1.0)]

    def datagen_multiple_slot_values():
        return  {'p_log': [1, 0.5, 0.7],
                'r': [1, 2, 3],
                'p_pred': [1, 0.7, 0.5]}

    def datagen_single_slot_value():
        return  {'p_log': [1],
                'r': [1],
                'p_pred': [1]}

    estimates_multiple = Helper.get_estimate(datagen_multiple_slot_values, listofestimators=[l[0] for l in listofestimators], num_examples=4)
    estimates_single = Helper.get_estimate(datagen_single_slot_value, listofestimators=[l[0] for l in listofestimators], num_examples=4)

    for Estimator, estimate_multiple, estimate_single in zip(listofestimators, estimates_multiple, estimates_single):
        Helper.assert_is_close(Estimator[1], estimate_multiple)
        Helper.assert_is_close(Estimator[1], estimate_single)
        assert estimate_single == estimate_multiple

def test_narrowing_intervals():
    ''' To test for narrowing intervals; Number of examples increase => narrowing CI '''

    listofintervals = [first_slot.Interval(cressieread.Interval()), first_slot.Interval(gaussian.Interval()), first_slot.Interval(clopper_pearson.Interval())]

    def datagen(epsilon, delta=0.5):
        # Logged Policy
        # 0 - (1-epsilon) : Reward is Bernoulli(delta)
        # 1 - epsilon : Reward is Bernoulli(1-delta)

        # p_pred: 1 if action is chosen, 0 if action not chosen

        # policy to estimate
        # (delta), (1-delta) reward from a Bernoulli distribution - for probability p_pred

        chosen = int(random.random() < epsilon)
        return {'p_log': [epsilon if chosen == 1 else 1 - epsilon],
                'r': [int(random.random() < 1-delta) if chosen == 1 else int(random.random() < delta)],
                'p_pred': [int(chosen==1)]}

    intervals_n1 = Helper.get_estimate(lambda: datagen(epsilon=0.5), listofintervals, num_examples=100)
    intervals_n2 = Helper.get_estimate(lambda: datagen(epsilon=0.5), listofintervals, num_examples=10000)

    for interval_n1, interval_n2 in zip(intervals_n1, intervals_n2):
        width_n1 = interval_n1[1] - interval_n1[0]
        width_n2 = interval_n2[1] - interval_n2[0]
        assert width_n1 > 0
        assert width_n2 > 0
        assert width_n2 < width_n1
