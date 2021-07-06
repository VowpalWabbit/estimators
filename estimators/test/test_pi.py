import os, sys, random, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimators.slates import pseudo_inverse
from estimators.bandits import ips
from estimators.bandits import snips
from estimators.bandits import mle
from estimators.bandits import cressieread
from estimators.bandits import cats_utils
from estimators.bandits import gaussian
from estimators.bandits import clopper_pearson

def test_bandits_unit_test():
    listofestimators = [(ips.Estimator(), 2.0), (snips.Estimator(), 1.0), (mle.Estimator(), 1.0), (cressieread.Estimator(), 1.0)]
    
    p_log = 0.3
    p_pred = 0.6
    reward = 1

    for Estimator in listofestimators:
        Estimator[0].add_example(p_log, reward, p_pred)
        assert Estimator[0].get() == Estimator[1]


def test_single_slot_pi_equivalent_to_ips():
    """PI should be equivalent to IPS when there is only a single slot"""
    pi_estimator = pseudo_inverse.Estimator()
    ips_estimator = ips.Estimator()
    is_close = lambda a, b: abs(a - b) <= 1e-6 * (1 + abs(a) + abs(b))

    p_logs = [0.8, 0.25, 0.5, 0.2]
    p_preds = [0.6, 0.4, 0.3, 0.9]
    rewards = [0.1, 0.2, 0, 1.0]

    for p_log, r, p_pred in zip(p_logs, rewards, p_preds):
        pi_estimator.add_example([p_log], r, [p_pred])
        ips_estimator.add_example(p_log, r, p_pred)
        assert is_close(pi_estimator.get() , ips_estimator.get())

def example_generator1():
    return  {'p_log': 1,
            'r': 1,
            'p_pred': 1}

def example_generator2(i, epsilon, p_pred=0.5):
    # Logged Policy
    # 0 - (1-epsilon) : Reward is always zero
    # 1 - epsilon : Reward is always 1

    # policy to estimate
    # 0, 1 reward- for probability p_pred

    chosen = int(random.random() < epsilon)
    return {'p_log': epsilon if chosen == 1 else 1 - epsilon,
            'r': 1 if chosen == 1 else 0,
            'p_pred':p_pred}

def example_generator3(i, epsilon, delta=0.5):
    # Logged Policy
    # 0 - (1-epsilon) : Reward is Bernoulli(delta)
    # 1 - epsilon : Reward is Bernoulli(1-delta)

    # p_pred: 1 if action is chosen, 0 if action not chosen

    chosen = int(random.random() < epsilon)
    return {'p_log': epsilon if chosen == 1 else 1 - epsilon,
            'r': int(random.random() < 1-delta) if chosen == 1 else int(random.random() < delta),
            'p_pred': int(chosen==1)}

def run_estimator(function, listofestimators, num_examples):
    is_close = lambda a, b: abs(a - b) <= 1e-6 * (1 + abs(a) + abs(b))
    for Estimator in listofestimators:
        # Estimator is a tuple
        # Estimator[0] is object of class Estimator()
        # Estimator[1] is expected value of the estimator
        for index in range(0,num_examples):
            data = function()
            Estimator[0].add_example(p_log=data['p_log'], r=data['r'], p_pred=data['p_pred'])
        assert is_close(Estimator[0].get(), Estimator[1])

def run_interval(function, listofintervals, n1, n2):
    """ n1 is the smaller than n2; Number of examples increase => narrowing CI"""

    datagen = lambda i: function(i, epsilon=0.5)

    for interval in listofintervals:

        # For n1 number of examples
        interval_n1 = copy.deepcopy(interval)
        for i in range(n1):
            data = datagen(i)
            interval_n1.add_example(p_log=data['p_log'], r=data['r'], p_pred=data['p_pred'])
        result_n1 = interval_n1.get()
        CI_n1 = abs(result_n1[1]-result_n1[0])

        # For n2 number of examples
        interval_n2 = copy.deepcopy(interval)
        for i in range(n2):
            data = datagen(i)
            interval_n2.add_example(p_log=data['p_log'], r=data['r'], p_pred=data['p_pred'])
        result_n2 = interval_n2.get()
        CI_n2 = abs(result_n2[1]-result_n2[0])

        assert (CI_n2 - CI_n1) < 0

def test_bandits():
    # The tuple (Estimator, expected value) for each estimator is stored in listofestimators
    listofestimators = [(ips.Estimator(), 1), (snips.Estimator(), 1), (mle.Estimator(), 1), (cressieread.Estimator(), 1)]
    run_estimator(example_generator1, listofestimators, 4)


def test_intervals():
    """ To test for narrowing intervals """
    listofintervals = [cressieread.Interval(), gaussian.Interval(), clopper_pearson.Interval()]
    run_interval(example_generator2, listofintervals, 100, 10000)
    run_interval(example_generator3, listofintervals, 100, 10000)


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
