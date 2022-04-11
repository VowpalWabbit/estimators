import numpy as np

from estimators import bandits
from estimators.ccb import first_slot, pdis_cressieread
from estimators.test.utils import Helper, Scenario, get_intervals


def assert_is_within(value, interval):
    if isinstance(value, list):
        assert len(value) == 2
        assert value[0] >= interval[0]
        assert value[1] <= interval[1]


def assert_more_examples_tighter_intervals(estimator, simulator):
    less_data = Scenario(lambda: simulator(100), estimator())
    more_data = Scenario(lambda: simulator(10000), estimator())

    less_data.get_interval()
    more_data.get_interval()

    assert len(less_data.result) == len(more_data.result)
    for i in range(len(less_data.result)):
        assert_is_within(more_data.result[i], less_data.result[i]) 


def test_more_examples_tighter_intervals():
    ''' To test if confidence intervals are getting tighter with more data points '''
    def simulator(n):
        for i in range(n):
            chosen = i % 2   
            yield  {'p_logs': [0.5, 1],
                    'rs': [chosen, (chosen + 1) % 2], 
                    'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    assert_more_examples_tighter_intervals(lambda: first_slot.Interval(bandits.cressieread.Interval()), simulator)
    assert_more_examples_tighter_intervals(lambda: first_slot.Interval(bandits.gaussian.Interval()), simulator)
    assert_more_examples_tighter_intervals(lambda: first_slot.Interval(bandits.clopper_pearson.Interval()), simulator)
    assert_more_examples_tighter_intervals(pdis_cressieread.Interval, simulator)


def assert_estimations_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_estimate()
    assert len(scenario.result) == len(expected)
    for i in range(len(expected)):
        assert_is_within(scenario.result[i], expected[i])


def test_estimations_convergence_simple():
    def simulator():
        for i in range(1000):
            chosen0 = i % 2   
            yield  {'p_logs': [0.5, 1],
                    'rs': [chosen0, chosen0], 
                    'p_preds': [0.2 if chosen0 == 1 else 0.8, 1]}

    expected = [(0.15, 0.25), (0.15, 0.25)]

    assert_estimations_within(pdis_cressieread.Estimator, simulator, expected)


def assert_intervals_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_interval()
    assert len(scenario.result) == len(expected)
    for i in range(len(expected)):
        assert_is_within(scenario.result[i], expected[i])


def test_interval_convergence_simple():
    def simulator():
        for i in range(1000):
            chosen0 = i % 2   
            yield  {'p_logs': [0.5, 1],
                    'rs': [chosen0, chosen0], 
                    'p_preds': [0.2 if chosen0 == 1 else 0.8, 1]}

    expected = [(0.15, 0.25), (0.15, 0.25)]

    assert_intervals_within(pdis_cressieread.Interval, simulator, expected)


def assert_higher_alpha_tighter_intervals(estimator, simulator):
    alphas = np.arange(0.1, 1, 0.1)

    scenarios = [Scenario(simulator, estimator(), alpha=alpha) for alpha in alphas]
    get_intervals(scenarios)

    for i in range(len(scenarios) - 1):
        assert len(scenarios[i].result) == len(scenarios[i + 1].result)
        for j in range(len(scenarios[i].result)):
            assert_is_within(scenarios[i].result[j], scenarios[i + 1].result[j])


def test_higher_alpha_tighter_intervals():
    ''' Get confidence intervals for various alpha levels and assert that they are shrinking as alpha increases'''
    def simulator():
        for i in range(1000):
            chosen = i % 2   
            yield  {'p_logs': [0.5, 1],
                    'rs': [chosen, (chosen + 1) % 2], 
                    'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    assert_higher_alpha_tighter_intervals(lambda: first_slot.Interval(bandits.cressieread.Interval()), simulator)
    assert_higher_alpha_tighter_intervals(lambda: first_slot.Interval(bandits.gaussian.Interval()), simulator)
    assert_higher_alpha_tighter_intervals(lambda: first_slot.Interval(bandits.clopper_pearson.Interval()), simulator)
    assert_higher_alpha_tighter_intervals(pdis_cressieread.Interval, simulator)


def test_various_slots_count():
    def simulator():
        for i in range(100):
            yield {'p_logs': [1, 1],
                   'rs': [1, 1],
                   'p_preds': [1, 1]}
            yield {'p_logs': [1],
                   'rs': [1],
                   'p_preds': [1]}

    expected = [(0.9, 1.1), (0.4, 0.6)]
    assert_estimations_within(pdis_cressieread.Estimator, simulator, expected)
    assert_intervals_within(pdis_cressieread.Interval, simulator, expected)


def test_convergence_with_no_overflow():
    def simulator():
        for i in range(1000000):
            chosen0 = i % 2   
            yield  {'p_logs': [0.5, 1],
                    'rs': [chosen0, chosen0], 
                    'p_preds': [0.2 if chosen0 == 1 else 0.8, 1]}

    expected = [(0.15, 0.25), (0.15, 0.25)]
    
    assert_estimations_within(pdis_cressieread.Estimator, simulator, expected)
    assert_intervals_within(pdis_cressieread.Interval, simulator, expected)


def test_no_data_estimation_is_none():
    assert first_slot.Estimator(bandits.ips.Estimator()).get() == []
    assert first_slot.Estimator(bandits.cressieread.Interval()).get() == []
    assert pdis_cressieread.Estimator().get() == []
    assert pdis_cressieread.Interval().get() == []

def assert_summation_works(estimator, simulator):
    scenario1000 = Scenario(lambda: simulator(1000), estimator())
    scenario2000 = Scenario(lambda: simulator(2000), estimator())
    scenario3000 = Scenario(lambda: simulator(3000), estimator())

    scenario1000.aggregate()
    scenario2000.aggregate()
    scenario3000.aggregate()

    result_1000_plus_2000 = (scenario1000.estimator + scenario2000.estimator).get()
    result_3000 = scenario3000.estimator.get()

    assert len(result_1000_plus_2000) == len(result_3000)
    for i in range(len(result_3000)):
        if isinstance(result_1000_plus_2000[i], float):
            Helper.assert_is_close(result_3000[i], result_1000_plus_2000[i])
        else:
            Helper.assert_is_close(result_3000[i][0], result_1000_plus_2000[i][0])     
            Helper.assert_is_close(result_3000[i][1], result_1000_plus_2000[i][1])    


def test_summation_works():
    def simulator(n):
        for i in range(n):
            chosen0 = i % 2   
            yield  {'p_logs': [0.5, 1],
                    'rs': [chosen0, chosen0], 
                    'p_preds': [0.2 if chosen0 == 1 else 0.8, 1]}

    assert_summation_works(pdis_cressieread.Estimator, simulator)
    assert_summation_works(pdis_cressieread.Interval, simulator)


def assert_summation_with_different_simulators_works(estimator, simulator1, simulator2, expected):
    scenario1 = Scenario(simulator1, estimator())
    scenario2 = Scenario(simulator2, estimator())

    scenario1.aggregate()
    scenario2.aggregate()

    result_1_plus_2 = (scenario1.estimator + scenario2.estimator).get()

    assert len(result_1_plus_2) == len(expected)
    for i in range(len(result_1_plus_2)):
        assert_is_within(result_1_plus_2[i], expected[i])


def test_summation_with_various_slots_works():
    def simulator1():
        for i in range(100):
            yield {'p_logs': [1, 1],
                   'rs': [1, 1],
                   'p_preds': [1, 1]}

    def simulator2():
        for i in range(100):
            yield {'p_logs': [1],
                   'rs': [1],
                   'p_preds': [1]}

    expected = [(0.9, 1.1), (0.4, 0.6)]
    assert_summation_with_different_simulators_works(pdis_cressieread.Estimator, simulator1, simulator2, expected)
    assert_summation_with_different_simulators_works(pdis_cressieread.Interval, simulator1, simulator2, expected)



