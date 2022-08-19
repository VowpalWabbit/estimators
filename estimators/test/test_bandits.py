import numpy as np

from estimators.bandits import ips
from estimators.bandits import snips
from estimators.bandits import mle
from estimators.bandits import cressieread
from estimators.bandits import cats_utils
from estimators.bandits import gaussian
from estimators.bandits import clopper_pearson
from estimators.bandits import cs
from estimators.test.utils import Helper, Scenario, get_intervals


def assert_estimation_is_close(estimator, simulator, value):
    scenario = Scenario(simulator, estimator())
    scenario.get_estimate()
    Helper.assert_is_close(scenario.result, value)

def assert_interval_covers(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_interval()
    assert scenario.result[0] <= expected
    assert scenario.result[1] >= expected


def test_estimate_1_from_1s():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''
    def simulator():
        for _ in range(10):
            yield  {'p_log': 1,
                    'r': 1,
                    'p_pred': 1}

    assert_estimation_is_close(ips.Estimator, simulator, 1)
    assert_estimation_is_close(snips.Estimator, simulator, 1)
    assert_estimation_is_close(mle.Estimator, simulator, 1)
    assert_estimation_is_close(cressieread.Estimator, simulator, 1)
    assert_interval_covers(gaussian.Interval, simulator, 1)
    assert_interval_covers(clopper_pearson.Interval, simulator, 1)
    assert_interval_covers(cressieread.Interval, simulator, 1)
    assert_interval_covers(cs.Interval, simulator, 1)


def test_estimate_10_from_10s():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''
    def simulator():
        for _ in range(10):
            yield  {'p_log': 1,
                    'r': 10,
                    'p_pred': 1}

    assert_estimation_is_close(ips.Estimator, simulator, 10)
    assert_estimation_is_close(snips.Estimator, simulator, 10)
    assert_estimation_is_close(mle.Estimator, simulator, 10)
    assert_estimation_is_close(cressieread.Estimator, simulator, 10)
    assert_interval_covers(gaussian.Interval, simulator, 10)
    assert_interval_covers(lambda: clopper_pearson.Interval(rmin=0, rmax=10), simulator, 10)
    assert_interval_covers(lambda: cressieread.Interval(rmin=0, rmax=10), simulator, 10)

def test_estimate_negative_constant():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''
    def simulator():
        for _ in range(10):
            yield  {'p_log': 1,
                    'r': -1,
                    'p_pred': 1}

    assert_estimation_is_close(ips.Estimator, simulator, -1)
    assert_estimation_is_close(snips.Estimator, simulator, -1)
    assert_estimation_is_close(mle.Estimator, simulator, -1)
    assert_estimation_is_close(cressieread.Estimator, simulator, -1)

    assert_interval_covers(gaussian.Interval, simulator, -1)
    assert_interval_covers(lambda: clopper_pearson.Interval(rmin=-1, rmax=0), simulator, -1)
    assert_interval_covers(lambda: cressieread.Interval(rmin=-1, rmax=0), simulator, -1)
    assert_interval_covers(lambda: cs.Interval(rmin=-1, rmax=0), simulator, -1)


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
            chosen = i % 2
            yield  {'p_log': 0.5,
                    'r': 1 if chosen == 0 else 0,
                    'p_pred': 0.2 if chosen == 0 else 0.8}

    assert_more_examples_tighter_intervals(cressieread.Interval, simulator)
    assert_more_examples_tighter_intervals(gaussian.Interval, simulator)
    assert_more_examples_tighter_intervals(clopper_pearson.Interval, simulator)
    assert_more_examples_tighter_intervals(cs.Interval, simulator)             


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
            chosen = i % 2
            yield  {'p_log': 0.5,
                    'r': 1 if chosen == 0 else 0,
                    'p_pred': 0.2 if chosen == 0 else 0.8}

    assert_higher_alpha_tighter_intervals(cressieread.Interval, simulator)
    assert_higher_alpha_tighter_intervals(gaussian.Interval, simulator)
    assert_higher_alpha_tighter_intervals(clopper_pearson.Interval, simulator)
    assert_higher_alpha_tighter_intervals(cs.Interval, simulator)     


def assert_interval_within(estimator, simulator, expected):
    scenario = Scenario(simulator, estimator())
    scenario.get_interval()
    assert scenario.result[0] >= expected[0]
    assert scenario.result[1] <= expected[1]


def assert_estimation_is_none(estimator):
    assert estimator().get() is None


def assert_interval_is_none(estimator):
    assert estimator().get()[0] is None
    assert estimator().get()[1] is None


def test_no_data_estimation_is_none():
    assert_estimation_is_none(ips.Estimator)
    assert_estimation_is_none(snips.Estimator)
    assert_estimation_is_none(mle.Estimator)
    assert_estimation_is_none(cressieread.Estimator)
    assert_interval_is_none(cressieread.Interval)
    assert_interval_is_none(gaussian.Interval)
    assert_interval_is_none(clopper_pearson.Interval)
    assert_interval_is_none(cs.Interval)


def test_simple_convergence():
    def simulator(r):
        for i in range(10000):
            chosen = i % 2
            yield  {'p_log': 0.5,
                    'r': r if chosen == 0 else 0,
                    'p_pred': 0.8 if chosen == 0 else 0.2}
    r = 1
    expected = (0.75 * r, 0.85 * r)
    assert_interval_within(gaussian.Interval, lambda: simulator(r), expected)
    assert_interval_within(lambda: clopper_pearson.Interval(0, r), lambda: simulator(r), expected)
    assert_interval_within(lambda: cressieread.Interval(rmin=0, rmax=r), lambda: simulator(r), expected)

    r = 10
    expected = (0.75 * r, 0.85 * r)
    assert_interval_within(gaussian.Interval, lambda: simulator(r), expected)
    assert_interval_within(lambda: clopper_pearson.Interval(0, r), lambda: simulator(r), expected)
    assert_interval_within(lambda: cressieread.Interval(rmin=0, rmax=r), lambda: simulator(r), expected)
    assert_interval_within(lambda: cs.Interval(rmin=0, rmax=r), lambda: simulator(r), expected)
   
    # TODO: test + fix for cressieread with negative rewards

def test_convergence_with_no_overflow():
    def simulator():
        for i in range(1000000):
            chosen = i % 2
            yield  {'p_log': 0.5,
                    'r': 1 if chosen == 0 else 0,
                    'p_pred': 0.2 if chosen == 0 else 0.8}
    expected = (0.15, 0.25)
    assert_interval_within(gaussian.Interval, simulator, expected)
    assert_interval_within(clopper_pearson.Interval, simulator, expected)
    assert_interval_within(cressieread.Interval, simulator, expected)
    assert_interval_within(cs.Interval, simulator, expected)


def assert_summation_works(estimator, simulator):
    scenario1000 = Scenario(lambda: simulator(1000), estimator())
    scenario2000 = Scenario(lambda: simulator(2000), estimator())
    scenario3000 = Scenario(lambda: simulator(3000), estimator())

    scenario1000.aggregate()
    scenario2000.aggregate()
    scenario3000.aggregate()

    result_1000_plus_2000 = (scenario1000.estimator + scenario2000.estimator).get()
    result_3000 = scenario3000.estimator.get()

    if isinstance(result_1000_plus_2000, float):
        Helper.assert_is_close(result_3000, result_1000_plus_2000)
    else:
        Helper.assert_is_close(result_3000[0], result_1000_plus_2000[0])     
        Helper.assert_is_close(result_3000[1], result_1000_plus_2000[1])    

def test_summation_works():
    def simulator(n):
        for i in range(n):
            chosen = i % 2
            yield  {'p_log': 0.5,
                    'r': 1 if chosen == 0 else 0,
                    'p_pred': 0.2 if chosen == 0 else 0.8}

    assert_summation_works(ips.Estimator, simulator)
    assert_summation_works(snips.Estimator, simulator)
    assert_summation_works(cressieread.Estimator, simulator)

    assert_summation_works(gaussian.Interval, simulator)
    assert_summation_works(clopper_pearson.Interval, simulator)
    assert_summation_works(cressieread.Interval, simulator)
    # TODO: fix
    #assert_summation_works(cs.Interval, simulator)

def test_convergence_with_count():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''
    def simulator():
        for _ in range(1000):
            yield  {'p_log': 1,
                    'r': 1,
                    'p_pred': 1}
        for _ in range(500):
            yield { 'p_log': 1,
                    'r': 0,
                    'p_pred': 1,
                    'count': 2
            }

    assert_estimation_is_close(ips.Estimator, simulator, 0.5)
    assert_estimation_is_close(snips.Estimator, simulator, 0.5)
    assert_estimation_is_close(mle.Estimator, simulator, 0.5)
    assert_estimation_is_close(cressieread.Estimator, simulator, 0.5)
    assert_interval_covers(gaussian.Interval, simulator, 0.5)
    assert_interval_covers(clopper_pearson.Interval, simulator, 0.5)
    assert_interval_covers(cressieread.Interval, simulator, 0.5)
    assert_interval_covers(cs.Interval, simulator, 0.5)

def test_convergence_with_p_drop():
    ''' To test correctness of estimators: Compare the expected value with value returned by Estimator.get()'''
    def simulator():
        for _ in range(1000):
            yield  {'p_log': 1,
                    'r': 1,
                    'p_pred': 1,
                    'p_drop': 0}
        for _ in range(500):
            yield { 'p_log': 1,
                    'r': 0,
                    'p_pred': 1,
                    'p_drop': 0.5
            }

    assert_interval_covers(gaussian.Interval, simulator, 0.5)
    assert_interval_covers(clopper_pearson.Interval, simulator, 0.5)
    assert_interval_covers(cressieread.Interval, simulator, 0.5)
    assert_interval_covers(cs.Interval, simulator, 0.5)

def assert_smaller_pdrop_tighter_intervals(estimator, simulator):
    small_pdrop = Scenario(lambda: simulator(0.1), estimator())
    big_pdrop = Scenario(lambda: simulator(0.5), estimator())

    small_pdrop.get_interval()
    big_pdrop.get_interval()

    assert big_pdrop.result[0] < small_pdrop.result[0]
    assert big_pdrop.result[1] > small_pdrop.result[1]    


def test_smaller_pdrop_tighter_intervals():
    def simulator(pdrop):
        for i in range(1000):
            chosen = i % 2
            yield  {'p_log': 0.5,
                    'r': 1 if chosen == 0 else 0,
                    'p_pred': 0.2 if chosen == 0 else 0.8,
                    'p_drop': pdrop}

    assert_smaller_pdrop_tighter_intervals(cressieread.Interval, simulator)
    assert_smaller_pdrop_tighter_intervals(gaussian.Interval, simulator)
    assert_smaller_pdrop_tighter_intervals(clopper_pearson.Interval, simulator)
    assert_smaller_pdrop_tighter_intervals(cs.Interval, simulator)               

def test_cats_ips():
    ips_estimator = ips.Estimator()
    snips_estimator = snips.Estimator()

    prob_logs = [0.151704, 0.006250, 0.086, 0.086, 0.086]
    action_logs = [15.0, 3.89, 22.3, 17.34, 31]
    rewards = [0.1, 0.2, 0, 1.0, 1.0]

    max_value = 32
    bandwidth = 1
    cats_transformer = cats_utils.CatsTransformer(num_actions=8, min_value=0, max_value=max_value, bandwidth=bandwidth)

    for logged_action, r, logged_prob in zip(action_logs, rewards, prob_logs):
        data = {}
        data['a'] = logged_action
        data['cost'] = r
        data['p'] = logged_prob
        if logged_action < (max_value / 2.0):
            pred_action = logged_action + 2 * bandwidth
            data = cats_transformer.transform(data, pred_action) # pred_action should be too far away, so pred_p should be 0
            assert data['pred_p'] == 0.0
        else:
            pred_action = logged_action
            data = cats_transformer.transform(data, logged_action) # same action, so pred_p should be 1
            assert data['pred_p'] == 1.0 / (2 * bandwidth)

        ips_estimator.add_example(data['p'], r, data['pred_p'])
        snips_estimator.add_example(data['p'], r, data['pred_p'])
    assert ips_estimator.get() >= snips_estimator.get()


def test_cats_transformer_on_edges():
    prob_logs = [0.151704, 0.006250, 0.086, 0.086]
    action_logs = [0, 1, 31, 32]
    rewards = [1.0, 1.0, 1.0, 1.0]

    max_value = 32
    bandwidth = 2
    cats_transformer = cats_utils.CatsTransformer(num_actions=8, min_value=0, max_value=max_value, bandwidth=bandwidth)

    for logged_action, r, logged_prob in zip(action_logs, rewards, prob_logs):
        data = {}
        data['a'] = logged_action
        data['cost'] = r
        data['p'] = logged_prob

        pred_action = logged_action
        data = cats_transformer.transform(data, logged_action) # same action, so pred_p should be 1
        assert data['pred_p'] == 1.0 / (2 * bandwidth)


def test_cats_baseline():
    max_value = 32
    min_value = 0
    bandwidth = 1
    num_actions = 8
    cats_transformer = cats_utils.CatsTransformer(num_actions=num_actions, min_value=min_value, max_value=max_value, bandwidth=bandwidth)
    baseline = cats_transformer.get_baseline1_prediction()
    ## unit range is 4, min_value is 0 so baseline action should be the centre of the firt unit range, starting off from min_value i.e. 2
    assert baseline == 2

    max_value = 33
    min_value = 1
    bandwidth = 1
    num_actions = 8
    cats_transformer = cats_utils.CatsTransformer(num_actions=num_actions, min_value=min_value, max_value=max_value, bandwidth=bandwidth)
    baseline = cats_transformer.get_baseline1_prediction()
    ## unit range is 4, min_value is 1 so baseline action should be the centre of the firt unit range, starting off from min_value i.e. 3
    assert baseline == 3
