import numpy as np

from estimators.ccb import multislot
from estimators.test.utils import Scenario, get_r_intervals, get_r_all_intervals


def assert_is_within(value, interval):
    if isinstance(value, list):
        assert len(value) == 2
        assert value[0] >= interval[0]
        assert value[1] <= interval[1]
    else:
        assert value >= interval[0]
        assert value <= interval[1]  


def assert_estimate_and_interval_convergence_after_swapping_slot_ids(
        estimator,
        interval_estimator,
        simulator,
        swap_slot_ids_simulator,
        expected_estimate,
        expected_estimate_after_swap):
    estimate_data = Scenario(lambda: simulator(1000, ['0', '1']), estimator())
    estimate_swap_slot_ids_data = Scenario(lambda: swap_slot_ids_simulator(1000), estimator())

    interval_data = Scenario(lambda: simulator(1000, ['0', '1']), interval_estimator(empirical_r_bounds=True))
    interval_swap_slot_ids_data = Scenario(lambda: swap_slot_ids_simulator(1000), interval_estimator(empirical_r_bounds=True))

    estimate_data.get_r_estimate()
    estimate_swap_slot_ids_data.get_r_estimate()

    interval_data.get_r_interval()
    interval_swap_slot_ids_data.get_r_interval()

    assert len(estimate_data.result) == len(expected_estimate)
    for i in range(len(expected_estimate)):
        assert_is_within(estimate_data.result[str(i)], expected_estimate[i])

    assert len(estimate_swap_slot_ids_data.result) == len(expected_estimate_after_swap)
    for i in range(len(expected_estimate_after_swap)):
        assert_is_within(estimate_swap_slot_ids_data.result[str(i)], expected_estimate_after_swap[i])

    assert len(interval_data.result) == len(expected_estimate)
    for i in range(len(expected_estimate)):
        assert_is_within(interval_data.result[str(i)], expected_estimate[i])

    assert len(interval_swap_slot_ids_data.result) == len(expected_estimate_after_swap)
    for i in range(len(expected_estimate_after_swap)):
        assert_is_within(interval_swap_slot_ids_data.result[str(i)], expected_estimate_after_swap[i])


def test_estimate_and_interval_convergence_after_swapping_slot_ids():
    def simulator(n, slot_ids):
        for i in range(n):
            chosen = i % 2
            yield {'slot_ids': slot_ids,
                   'p_logs': [0.5, 1],
                   'rs': [chosen, (chosen + 1) % 2],
                   'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    def swap_slot_ids_simulator(n):
        for ex in simulator(n, ['0', '1']):
            yield ex
        for ex in simulator(2 * n, ['1', '0']):
            yield ex

    expected_slot_0_estimate = 0.2
    expected_slot_1_estimate = 0.8
    expected_estimate = [(expected_slot_0_estimate - 0.05, expected_slot_0_estimate + 0.05),
                         (expected_slot_1_estimate - 0.05, expected_slot_1_estimate + 0.05)]

    expected_slot_0_estimate_after_swap = (1 * expected_slot_0_estimate  + 2 * expected_slot_1_estimate) / (1 + 2)
    expected_slot_1_estimate_after_swap = (1 * expected_slot_1_estimate  + 2 * expected_slot_0_estimate) / (1 + 2)
    expected_estimate_after_swap = [(expected_slot_0_estimate_after_swap - 0.05, expected_slot_0_estimate_after_swap + 0.05),
                                    (expected_slot_1_estimate_after_swap - 0.05, expected_slot_1_estimate_after_swap + 0.05)]

    assert_estimate_and_interval_convergence_after_swapping_slot_ids(
        multislot.Estimator,
        multislot.Interval,
        simulator,
        swap_slot_ids_simulator,
        expected_estimate,
        expected_estimate_after_swap)


def assert_estimates_within_interval_bounds(estimator, interval_estimator, simulator):
    estimate_data = Scenario(lambda: simulator(100), estimator())
    interval_data = Scenario(lambda: simulator(100), interval_estimator(empirical_r_bounds=True))

    estimate_data.get_r_estimate()
    interval_data.get_r_interval()

    assert len(estimate_data.result) == len(interval_data.result)
    for key in estimate_data.result:
        assert_is_within(estimate_data.result[key], interval_data.result[key])

def assert_estimates_all_within_interval_bounds(estimator, interval_estimator, simulator):
    estimate_data = Scenario(lambda: simulator(100), estimator())
    interval_data = Scenario(lambda: simulator(100), interval_estimator(empirical_r_bounds=True))

    estimate_data.get_r_all_estimate()
    interval_data.get_r_all_interval()

    assert_is_within(estimate_data.result, interval_data.result)


def test_estimates_within_interval_bounds():
    ''' To test if estimates are within interval bounds '''
    def simulator(n):
        for i in range(n):
            chosen = i % 2
            yield {'slot_ids': ['0', '1'],
                   'p_logs': [0.5, 1],
                   'rs': [chosen, (chosen + 1) % 2],
                   'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    assert_estimates_within_interval_bounds(multislot.Estimator, multislot.Interval, simulator)
    assert_estimates_all_within_interval_bounds(multislot.Estimator, multislot.Interval, simulator)


def assert_more_examples_tighter_intervals(estimator, simulator):
    less_data = Scenario(lambda: simulator(100), estimator())
    more_data = Scenario(lambda: simulator(10000), estimator())

    less_data.get_r_interval()
    more_data.get_r_interval()

    assert len(less_data.result) == len(more_data.result)
    for key in less_data.result:
        assert_is_within(more_data.result[key], less_data.result[key])


def assert_more_examples_tighter_intervals_all(estimator, simulator):
    less_data = Scenario(lambda: simulator(100), estimator())
    more_data = Scenario(lambda: simulator(10000), estimator())

    less_data.get_r_all_interval()
    more_data.get_r_all_interval()

    assert_is_within(more_data.result, less_data.result)


def test_more_examples_tighter_intervals():
    ''' To test if confidence intervals are getting tighter with more data points '''
    def simulator(n):
        for i in range(n):
            chosen = i % 2
            yield {'slot_ids': ['0', '1'],
                   'p_logs': [0.5, 1],
                   'rs': [chosen, (chosen + 1) % 2],
                   'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    assert_more_examples_tighter_intervals(multislot.Interval, simulator)
    assert_more_examples_tighter_intervals_all(multislot.Interval, simulator)


def assert_estimations_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_r_estimate()
    assert len(scenario.result) == len(expected)
    for i in range(len(expected)):
        assert_is_within(scenario.result[str(i)], expected[i])


def assert_estimations_all_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_r_all_estimate()
    assert_is_within(scenario.result, expected)


def test_estimations_convergence_simple():
    def simulator():
        for i in range(1000):
            chosen0 = i % 2
            yield {'slot_ids': ['0', '1'],
                   'p_logs': [0.5, 1],
                   'rs': [chosen0, chosen0],
                   'p_preds': [0.2 if chosen0 == 1 else 0.8, 1]}

    expected = [(0.15, 0.25), (0.15, 0.25)]
    assert_estimations_within(multislot.Estimator, simulator, expected)

    expected = (0.15, 0.25)
    assert_estimations_all_within(multislot.Estimator, simulator, expected)


def assert_intervals_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_r_interval()
    assert len(scenario.result) == len(expected)
    for i in range(len(expected)):
        assert_is_within(scenario.result[str(i)], expected[i])


def assert_intervals_all_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_r_all_interval()
    assert_is_within(scenario.result, expected)


def test_interval_convergence_simple():
    def simulator():
        for i in range(1000):
            chosen0 = i % 2
            yield {'slot_ids': ['0', '1'],
                   'p_logs': [0.5, 1],
                   'rs': [chosen0, chosen0],
                   'p_preds': [0.2 if chosen0 == 1 else 0.8, 1]}

    expected = [(0.15, 0.25), (0.15, 0.25)]
    assert_intervals_within(multislot.Interval, simulator, expected)

    expected = (0.15, 0.25)
    assert_intervals_all_within(multislot.Interval, simulator, expected)


def assert_higher_alpha_tighter_intervals(estimator, simulator):
    alphas = np.arange(0.1, 1, 0.1)

    scenarios = [Scenario(simulator, estimator(), alpha=alpha) for alpha in alphas]
    get_r_intervals(scenarios)

    for i in range(len(scenarios) - 1):
        assert len(scenarios[i].result) == len(scenarios[i + 1].result)
        for j in range(len(scenarios[i].result)):
            assert_is_within(scenarios[i + 1].result[str(j)], scenarios[i].result[str(j)])


def assert_higher_alpha_tighter_intervals_all(estimator, simulator):
    alphas = np.arange(0.1, 1, 0.1)

    scenarios = [Scenario(simulator, estimator(), alpha=alpha) for alpha in alphas]
    get_r_all_intervals(scenarios)

    for i in range(len(scenarios) - 1):
        assert_is_within(scenarios[i + 1].result, scenarios[i].result)


def test_higher_alpha_tighter_intervals():
    ''' Get confidence intervals for various alpha levels and assert that they are shrinking as alpha increases'''
    def simulator():
        for i in range(1000):
            chosen = i % 2
            yield {'slot_ids': ['0', '1'],
                   'p_logs': [0.5, 1],
                   'rs': [chosen, (chosen + 1) % 2],
                   'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    assert_higher_alpha_tighter_intervals(multislot.Interval, simulator)
    assert_higher_alpha_tighter_intervals_all(multislot.Interval, simulator)


def test_various_slots_count():
    def simulator():
        for i in range(100):
            yield {'slot_ids': ['0', '1'],
                   'p_logs': [1, 1],
                   'rs': [1, 1],
                   'p_preds': [1, 1]}
            yield {'slot_ids': ['0'],
                   'p_logs': [1],
                   'rs': [1],
                   'p_preds': [1]}

    expected = [(0.9, 1.1), (0.4, 0.6)]
    assert_estimations_within(multislot.Estimator, simulator, expected)
    assert_intervals_within(multislot.Interval, simulator, expected)
    
    expected = (0.9, 1.1)
    assert_estimations_all_within(multislot.Estimator, simulator, expected)
    assert_intervals_all_within(multislot.Interval, simulator, expected)


def test_convergence_with_no_overflow():
    def simulator():
        for i in range(1000000):
            chosen0 = i % 2
            yield {'slot_ids': ['0', '1'],
                   'p_logs': [0.5, 1],
                   'rs': [chosen0, chosen0],
                   'p_preds': [0.2 if chosen0 == 1 else 0.8, 1]}

    expected = [(0.15, 0.25), (0.15, 0.25)]

    assert_estimations_within(multislot.Estimator, simulator, expected)
    assert_intervals_within(multislot.Interval, simulator, expected)

    expected = (0.15, 0.25)

    assert_estimations_all_within(multislot.Estimator, simulator, expected)
    assert_intervals_all_within(multislot.Interval, simulator, expected)


def test_no_data_estimation_is_none():
    assert multislot.Estimator().get_r() == {}
    assert multislot.Interval().get_r() == {}
    assert multislot.Estimator().get_r_all() == None
    assert multislot.Interval().get_r_all[0] == None
    assert multislot.Interval().get_r_all[1] == None


def assert_summation_with_different_simulators_works(estimator, simulator1, simulator2, expected):
    scenario1 = Scenario(simulator1, estimator())
    scenario2 = Scenario(simulator2, estimator())

    scenario1.aggregate()
    scenario2.aggregate()

    result_1_plus_2 = (scenario1.estimator + scenario2.estimator).get_r()

    assert len(result_1_plus_2) == len(expected)
    for id in result_1_plus_2.keys():
        assert_is_within(result_1_plus_2[id], expected[id])


def test_summation_with_various_slots_works():
    def simulator1():
        for i in range(100):
            yield {'slot_ids': ['0', '1'],
                   'p_logs': [1, 1],
                   'rs': [1, 1],
                   'p_preds': [1, 1]}

    def simulator2():
        for i in range(100):
            yield {'slot_ids': ['0', '2'],
                   'p_logs': [1, 1],
                   'rs': [1, 1],
                   'p_preds': [1, 1]}

    expected = {'0' : (0.9, 1.1), '1': (0.4, 0.6), '2': (0.4, 0.6)}
    assert_summation_with_different_simulators_works(multislot.Estimator, simulator1, simulator2, expected)
    assert_summation_with_different_simulators_works(multislot.Interval, simulator1, simulator2, expected)

