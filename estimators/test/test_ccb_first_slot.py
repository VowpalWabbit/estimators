from estimators.bandits import ips
from estimators.bandits import snips
from estimators.bandits import mle
from estimators.bandits import cressieread
from estimators.ccb import first_slot
from estimators.test.utils import Helper

# TODO: add test that first slot estimator is equivalent to corresponding cb estimator and remove it from all tests in test_ccb


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
    """To test correctness of estimators: Compare the expected value with value returned by Estimator.get()"""

    # The tuple (Estimator, expected value) for each estimator is stored in estimators
    estimators = [
        (first_slot.Estimator(ips.Estimator()), 1.0),
        (first_slot.Estimator(snips.Estimator()), 1.0),
        (first_slot.Estimator(mle.Estimator()), 1.0),
        (first_slot.Estimator(cressieread.Estimator()), 1.0),
    ]

    def datagen_multiple_slot_values():
        return {"p_log": [1, 0.5, 0.7], "r": [1, 2, 3], "p_pred": [1, 0.7, 0.5]}

    def datagen_single_slot_value():
        return {"p_log": [1], "r": [1], "p_pred": [1]}

    estimates_multiple = Helper.get_estimate(
        datagen_multiple_slot_values,
        estimators=[l[0] for l in estimators],
        num_examples=4,
    )
    estimates_single = Helper.get_estimate(
        datagen_single_slot_value, estimators=[l[0] for l in estimators], num_examples=4
    )

    for Estimator, estimate_multiple, estimate_single in zip(
        estimators, estimates_multiple, estimates_single
    ):
        Helper.assert_is_close(Estimator[1], estimate_multiple[0])
        Helper.assert_is_close(Estimator[1], estimate_single[0])
        assert estimate_single[0] == estimate_multiple[0]
