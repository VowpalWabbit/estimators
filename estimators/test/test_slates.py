import numpy as np

from estimators.slates import pseudo_inverse
from estimators.slates import gaussian
from estimators.test.utils import Helper, Scenario, get_intervals

def assert_estimation_is_close(estimator, simulator, value):
    scenario = Scenario(simulator, estimator())
    scenario.get_estimate()
    Helper.assert_is_close(scenario.result, value)


def test_estimate_1_from_1s():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''
    def simulator():
        for _ in range(10):
            yield  {'p_logs': [1] * 4,
                    'r': 1,
                    'p_preds': [1] * 4}

    assert_estimation_is_close(pseudo_inverse.Estimator, simulator, 1)


def assert_more_examples_tighter_intervals(estimator, simulator):
    less_data = Scenario(lambda: simulator(100), estimator())
    more_data = Scenario(lambda: simulator(10000), estimator())

    less_data.get_interval()
    more_data.get_interval()

    assert less_data.result[0] <= more_data.result[0]
    assert less_data.result[1] >= more_data.result[1]    


def test_more_examples_tighter_intervals():
    ''' To test if confidence intervals are getting tighter with more data points '''
    def simulator(n):
        for i in range(n):
            chosen = [i % 2, i % 4 // 2]
            rewards = [
                [0, 0],
                [0, 1]
            ]
            policy = [
                [0.8, 0.2],
                [0.8, 0.2]
            ]
            yield {'p_logs': [0.5, 0.5],
                   'r': rewards[chosen[0]][chosen[1]],
                   'p_preds': [policy[0][chosen[0]], policy[1][chosen[1]]]}

    assert_more_examples_tighter_intervals(gaussian.Interval, simulator)


def assert_higher_alpha_tighter_intervals(estimator, simulator):
    alphas = np.arange(0.1, 1, 0.1)

    scenarios = [Scenario(simulator, estimator(), alpha=alpha) for alpha in alphas]
    get_intervals(scenarios)

    for i in range(len(scenarios) - 1):
        assert scenarios[i].result[0] <= scenarios[i + 1].result[0]
        assert scenarios[i].result[1] >= scenarios[i + 1].result[1]


def test_higher_alpha_tighter_intervals():
    ''' Get confidence intervals for various alpha levels and assert that they are shrinking as alpha increases'''
    def simulator():
        for i in range(1000):
            chosen = [i % 2, i % 4 // 2]
            rewards = [
                [0, 0],
                [0, 1]
            ]
            policy = [
                [0.8, 0.2],
                [0.8, 0.2]
            ]
            yield {'p_logs': [0.5, 0.5],
                   'r': rewards[chosen[0]][chosen[1]],
                   'p_preds': [policy[0][chosen[0]], policy[1][chosen[1]]]}

    assert_higher_alpha_tighter_intervals(gaussian.Interval, simulator)


def test_no_data_estimation_is_none():
    assert gaussian.Interval().get() == [None, None]
    assert pseudo_inverse.Estimator().get() is None


def assert_interval_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_interval()
    assert scenario.result[0] >= expected[0]
    assert scenario.result[1] <= expected[1]


def test_convergence_simple():
    def simulator():
        for i in range(1000):
            chosen = [i % 2, i % 4 // 2]
            rewards = [
                [0, 0],
                [0, 1]
            ]
            policy = [
                [0.8, 0.2],
                [0.8, 0.2]
            ]
            yield {'p_logs': [0.5, 0.5],
                   'r': rewards[chosen[0]][chosen[1]],
                   'p_preds': [policy[0][chosen[0]], policy[1][chosen[1]]]}

        assert_estimation_is_close(pseudo_inverse.Estimator, simulator, 0.2)
        assert_interval_within(gaussian.Interval, simulator, (0.15, 0.25))
