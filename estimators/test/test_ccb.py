import random

from estimators.bandits import ips
from estimators.bandits import snips
from estimators.bandits import mle
from estimators.bandits import cressieread
from estimators.bandits import gaussian
from estimators.bandits import clopper_pearson
from estimators.ccb import first_slot
from estimators.test.utils import Helper

random.seed(0)
# TBD: rethink tests given proper estimator (with estimations from all slots added)
def test_single_example():
    estimators = [
        (first_slot.Estimator(ips.Estimator()), 2.0),
        (first_slot.Estimator(snips.Estimator()), 1.0),
        (first_slot.Estimator(mle.Estimator()), 1.0),
        (first_slot.Estimator(cressieread.Estimator()), 1.0),
        ]

    p_log = [0.3]
    p_pred = [0.6]
    reward = [1]

    for Estimator in estimators:
        Estimator[0].add_example(p_log, reward, p_pred)
        assert Estimator[0].get()[0] == Estimator[1]

def test_multiple_examples():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''

    # The tuple (Estimator, expected value) for each estimator is stored in estimators
    estimators = [
        (first_slot.Estimator(ips.Estimator()), 1.0),
        (first_slot.Estimator(snips.Estimator()), 1.0),
        (first_slot.Estimator(mle.Estimator()), 1.0),
        (first_slot.Estimator(cressieread.Estimator()), 1.0)
        ]

    def datagen_multiple_slot_values():
        return  {'p_log': [1, 0.5, 0.7],
                'r': [1, 2, 3],
                'p_pred': [1, 0.7, 0.5]}

    def datagen_single_slot_value():
        return  {'p_log': [1],
                'r': [1],
                'p_pred': [1]}

    estimates_multiple = Helper.get_estimate(datagen_multiple_slot_values, estimators=[l[0] for l in estimators], num_examples=4)
    estimates_single = Helper.get_estimate(datagen_single_slot_value, estimators=[l[0] for l in estimators], num_examples=4)

    for Estimator, estimate_multiple, estimate_single in zip(estimators, estimates_multiple, estimates_single):
        Helper.assert_is_close(Estimator[1], estimate_multiple[0])
        Helper.assert_is_close(Estimator[1], estimate_single[0])
        assert estimate_single[0] == estimate_multiple[0]

def test_narrowing_intervals():
    ''' To test if confidence intervals are getting tighter with more data points '''

    intervals = [
        first_slot.Interval(cressieread.Interval()),
        first_slot.Interval(gaussian.Interval()),
        first_slot.Interval(clopper_pearson.Interval()),
        ]

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

    intervals_less_data = Helper.get_estimate(lambda: datagen(epsilon=0.5), intervals, num_examples=100)
    intervals_more_data = Helper.get_estimate(lambda: datagen(epsilon=0.5), intervals, num_examples=10000)

    for interval_less_data, interval_more_data in zip(intervals_less_data, intervals_more_data):
        width_wider = interval_less_data[0][1] - interval_less_data[0][0]
        width_narrower = interval_more_data[0][1] - interval_more_data[0][0]
        assert width_wider > 0
        assert width_narrower > 0
        assert width_narrower < width_wider
