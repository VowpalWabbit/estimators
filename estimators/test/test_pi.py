import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slates import pseudo_inverse
from bandits import ips
from bandits import snips
from bandits import mle
from bandits import cressieread
from bandits import cats_utils

def test_ips():
    ips_estimator = ips.Estimator()

    p_log = 0.3
    p_pred = 0.6
    reward = 1

    ips_estimator.add_example(p_log, reward, p_pred)
    assert ips_estimator.get() == 2.0


def test_snips():
    snips_estimator = snips.Estimator()

    p_log = 0.3
    p_pred = 0.6
    reward = 1

    snips_estimator.add_example(p_log, reward, p_pred)
    assert snips_estimator.get() == 1.0


def test_mle():
    mle_estimator = mle.Estimator()

    p_log = 0.3
    p_pred = 0.6
    reward = 1

    mle_estimator.add_example(p_log, reward, p_pred)
    assert mle_estimator.get() == 1.0


def test_cressieread():
    cressieread_estimator = cressieread.Estimator()

    p_log = 0.3
    p_pred = 0.6
    reward = 1

    cressieread_estimator.add_example(p_log, reward, p_pred)
    assert cressieread_estimator.get() == 1.0


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
