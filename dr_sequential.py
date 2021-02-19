import os
import ds_parse
from vowpalwabbit import pyvw

class Estimator:
    def __init__(self):
        ############################### Aggregates quantities ######################################
        #
        # 'tot':   DRE sum value
        # 'N':   total number of samples in bin from log (IPS = n/N)
        #
        #################################################################################################

        self.data = {'tot':0.,'N':0}

    def add_example(self, x_feat_string, a, a_vec, p_log, r, p_pred, policy_probas, reward_estimator_model, count=1):
        self.data['N'] += count

        r_hat_xk_pi = sum(policy_probas[i] * reward_estimator_model.predict("|ANamespace Action:" + str(ac)+ x_feat_string) for i, ac in enumerate(a_vec))
        self.data['tot'] += r_hat_xk_pi

        p_over_p = p_pred/p_log
        rk_minus_r_hat = r - reward_estimator_model.predict("|ANamespace Action:" + str(a) + x_feat_string)

        self.data['tot'] += (p_over_p * rk_minus_r_hat)

    def get_estimate(self):
        est = 0

        return est

    def reward_estimator(self, train_data, reward_estimator_model, mode='batch', load_model=False):

        return reward_estimator_model
