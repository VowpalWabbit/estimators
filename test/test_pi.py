import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pseudo_inverse
import ips_snips
import cats_utils

def test_single_slot_pi_equivalent_to_ips():
    """PI should be equivalent to IPS when there is only a single slot"""
    pi_estimator = pseudo_inverse.Estimator()
    ips_estimator = ips_snips.Estimator()
    is_close = lambda a, b: abs(a - b) <= 1e-6 * (1 + abs(a) + abs(b))

    p_logs = [0.8, 0.25, 0.5, 0.2]
    p_preds = [0.6, 0.4, 0.3, 0.9]
    rewards = [0.1, 0.2, 0, 1.0]

    for p_log, r, p_pred in zip(p_logs, rewards, p_preds):
        pi_estimator.add_example([p_log], r, [p_pred])
        ips_estimator.add_example(p_log, r, p_pred)
        assert is_close(pi_estimator.get_estimate('pi') , ips_estimator.get_estimate('ips'))


def test_cats_ips():
    ips_estimator = ips_snips.Estimator()

    prob_logs = [0.151704, 0.006250, 0.086, 0.086]
    action_logs = [15.0, 3.89, 22.3, 17.34]
    rewards = [0.1, 0.2, 0, 1.0]

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
        assert ips_estimator.get_estimate('ips') >= ips_estimator.get_estimate('snips')


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
