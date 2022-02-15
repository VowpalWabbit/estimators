import numpy as np

from estimators.bandits import clopper_pearson
from estimators.bandits import gaussian
from estimators.bandits import cressieread
from estimators.ccb import first_slot
from estimators.test.utils import Helper, Scenario, get_intervals

def assert_more_examples_tighter_intervals(estimator, simulator):
    less_data = Scenario(lambda: simulator(100), estimator())
    more_data = Scenario(lambda: simulator(10000), estimator())

    less_data.get_interval()
    more_data.get_interval()

    assert len(less_data.result) == len(more_data.result)
    for i in range(len(less_data.result)):
        assert less_data.result[i][0] <= more_data.result[i][0]
        assert less_data.result[i][1] >= more_data.result[i][1]    

def test_more_examples_tighter_intervals():
    ''' To test if confidence intervals are getting tighter with more data points '''
    def simulator(n):
        for i in range(n):
            chosen = i % 2   
            yield  {'p_logs': [0.5, 1],
                    'rs': [chosen, (chosen + 1) % 2], 
                    'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    assert_more_examples_tighter_intervals(lambda: first_slot.Interval(cressieread.Interval()), simulator)
    assert_more_examples_tighter_intervals(lambda: first_slot.Interval(gaussian.Interval()), simulator)
    assert_more_examples_tighter_intervals(lambda: first_slot.Interval(clopper_pearson.Interval()), simulator)


def assert_higher_alpha_tighter_intervals(estimator, simulator):
    alphas = np.arange(0.1, 1, 0.1)

    scenarios = [Scenario(simulator, estimator(), alpha=alpha) for alpha in alphas]
    get_intervals(scenarios)

    for i in range(len(scenarios) - 1):
        assert len(scenarios[i].result) == len(scenarios[i + 1].result)
        for j in range(len(scenarios[i].result)):
            assert scenarios[i].result[j][0] <= scenarios[i + 1].result[j][0]
            assert scenarios[i].result[j][1] >= scenarios[i + 1].result[j][1]


def test_higher_alpha_tighter_intervals():
    ''' Get confidence intervals for various alpha levels and assert that they are shrinking as alpha increases'''
    def simulator():
        for i in range(1000):
            chosen = i % 2   
            yield  {'p_logs': [0.5, 1],
                    'rs': [chosen, (chosen + 1) % 2], 
                    'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    assert_higher_alpha_tighter_intervals(lambda: first_slot.Interval(cressieread.Interval()), simulator)
    assert_higher_alpha_tighter_intervals(lambda: first_slot.Interval(gaussian.Interval()), simulator)
    assert_higher_alpha_tighter_intervals(lambda: first_slot.Interval(clopper_pearson.Interval()), simulator)
