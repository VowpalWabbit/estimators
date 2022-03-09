import numpy as np

from estimators.ccb import multislot
from estimators.test.utils import Scenario, get_intervals


def assert_estimate_and_interval_convergence_after_swapping_slot_ids(
        estimator,
        interval_estimator,
        simulator,
        swap_slot_ids_simulator,
        expected_estimate,
        expected_estimate_after_swap):
    estimate_data = Scenario(lambda: simulator(1000), estimator())
    estimate_swap_slot_ids_data = Scenario(lambda: swap_slot_ids_simulator(3000), estimator())

    interval_data = Scenario(lambda: simulator(1000), interval_estimator())
    interval_swap_slot_ids_data = Scenario(lambda: swap_slot_ids_simulator(3000), interval_estimator())

    estimate_data.get_estimate()
    estimate_swap_slot_ids_data.get_estimate()

    interval_data.get_interval()
    interval_swap_slot_ids_data.get_interval()

    assert len(estimate_data.result) == len(expected_estimate)
    for i in range(len(expected_estimate)):
        assert estimate_data.result[str(i)] >= expected_estimate[i][0]
        assert estimate_data.result[str(i)] <= expected_estimate[i][1]

    assert len(estimate_swap_slot_ids_data.result) == len(expected_estimate_after_swap)
    for i in range(len(expected_estimate_after_swap)):
        assert estimate_swap_slot_ids_data.result[str(i)] >= expected_estimate_after_swap[i][0]
        assert estimate_swap_slot_ids_data.result[str(i)] <= expected_estimate_after_swap[i][1]

    assert len(interval_data.result) == len(expected_estimate)
    for i in range(len(expected_estimate)):
        assert interval_data.result[str(i)][0] >= expected_estimate[i][0]
        assert interval_data.result[str(i)][1] <= expected_estimate[i][1]

    assert len(interval_swap_slot_ids_data.result) == len(expected_estimate_after_swap)
    for i in range(len(expected_estimate_after_swap)):
        assert interval_swap_slot_ids_data.result[str(i)][0] >= expected_estimate_after_swap[i][0]
        assert interval_swap_slot_ids_data.result[str(i)][1] <= expected_estimate_after_swap[i][1]


def test_estimate_and_interval_convergence_after_swapping_slot_ids():
    def simulator(n):
        for i in range(n):
            chosen = i % 2
            yield {'slot_ids': ['0', '1'],
                   'p_logs': [0.5, 1],
                   'rs': [chosen, (chosen + 1) % 2],
                   'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    def swap_slot_ids_simulator(n):
        for i in range(n):
            chosen = i % 2
            if i < n/3:
                yield {'slot_ids': ['0', '1'],
                       'p_logs': [0.5, 1],
                       'rs': [chosen, (chosen + 1) % 2],
                       'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}
            else:
                yield {'slot_ids': ['1', '0'],
                       'p_logs': [0.5, 1],
                       'rs': [chosen, (chosen + 1) % 2],
                       'p_preds': [0.5 + 0.3 * (-1)**chosen, 1]}

    expected_slot_0_estimate = 0.2
    expected_slot_1_estimate = 0.8
    expected_estimate = [(expected_slot_0_estimate - 0.05, expected_slot_0_estimate + 0.05),
                         (expected_slot_1_estimate - 0.05, expected_slot_1_estimate + 0.05)]

    expected_slot_0_estimate_after_swap = 0.6
    expected_slot_1_estimate_after_swap = 0.4
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
    interval_data = Scenario(lambda: simulator(100), interval_estimator())

    estimate_data.get_estimate()
    interval_data.get_interval()

    assert len(estimate_data.result) == len(interval_data.result)
    for key in estimate_data.result:
        assert estimate_data.result[key] > interval_data.result[key][0]
        assert estimate_data.result[key] < interval_data.result[key][1]


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


def assert_more_examples_tighter_intervals(estimator, simulator):
    less_data = Scenario(lambda: simulator(100), estimator())
    more_data = Scenario(lambda: simulator(10000), estimator())

    less_data.get_interval()
    more_data.get_interval()

    assert len(less_data.result) == len(more_data.result)
    for key in less_data.result:
        assert less_data.result[key][0] <= more_data.result[key][0]
        assert less_data.result[key][1] >= more_data.result[key][1]


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


def assert_estimations_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_estimate()
    assert len(scenario.result) == len(expected)
    for i in range(len(expected)):
        assert scenario.result[str(i)] >= expected[i][0]
        assert scenario.result[str(i)] <= expected[i][1]


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


def assert_intervals_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_interval()
    assert len(scenario.result) == len(expected)
    for i in range(len(expected)):
        assert scenario.result[str(i)][0] >= expected[i][0]
        assert scenario.result[str(i)][1] <= expected[i][1]


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


def assert_higher_alpha_tighter_intervals(estimator, simulator):
    alphas = np.arange(0.1, 1, 0.1)

    scenarios = [Scenario(simulator, estimator(), alpha=alpha) for alpha in alphas]
    get_intervals(scenarios)

    for i in range(len(scenarios) - 1):
        assert len(scenarios[i].result) == len(scenarios[i + 1].result)
        for j in range(len(scenarios[i].result)):
            assert scenarios[i].result[str(j)][0] <= scenarios[i + 1].result[str(j)][0]
            assert scenarios[i].result[str(j)][1] >= scenarios[i + 1].result[str(j)][1]


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


def test_no_data_estimation_is_none():
    assert multislot.Estimator().get() == {}
    assert multislot.Interval().get() == {}
