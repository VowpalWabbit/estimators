import math
from scipy.stats import beta

# PseudoInverse estimator for slate recommendation. The following implements the
# case for a Cartesian product when mu is a product distribution. This can be
# seen in example 4 of the paper.
# https://arxiv.org/abs/1605.04812

class Estimator:
    def __init__(self):
        self.data = {'n':0.,'N':0, 'SoS':0}

    def add_example(self, p_logs, r, p_preds, count=1):
        """Expects lists for logged probabilities and predicted probabilities. These should correspond to each slot.
        This function is implemented under the simplifying assumptions of
        example 4 in the paper 'Off-policy evaluation for slate recommendation'
        where the slate space is a cartesian product and the logging policy is a
        product distribution"""
        if not isinstance(p_logs, list) or not isinstance(p_preds, list):
            raise('Error: p_logs and p_preds must be lists')

        if(len(p_logs) != len(p_preds)):
            raise('Error: p_logs and p_preds must be the same length, found {} and {} respectively'.format(len(p_logs), len(p_preds)))

        self.data['N'] += count
        p_over_ps = 0
        num_slots = len(p_logs)
        for p_log, p_pred in zip(p_logs, p_preds):
            p_over_ps += p_pred/p_log
        p_over_ps -= num_slots - 1

        if r != 0:
            self.data['n'] += r*p_over_ps*count
            self.data['SoS'] += ((r*p_over_ps)**2)*count

    def get_estimate(self, type):
        if self.data['N'] == 0:
            raise('Error: No data point added')

        if type == 'pi':
            return self.data['n']/self.data['N']
        else:
            raise('Error: Incorrect estimator type {}. Supported options are pi'.format(type))

    def get_interval(self, type, alpha=0.05):
        bounds = []
        num = self.data['n']
        den = self.data['N']
        SoS = self.data['SoS']

        if type == "gaussian":
            if SoS > 0.0:
                zGaussianCdf = {
                  0.25: 1.15,
                  0.1: 1.645,
                  0.05: 1.96
                }

                variance = (SoS - num * num / den) / (den - 1)
                gaussDelta = zGaussianCdf[alpha] * math.sqrt(variance/den)
                bounds.append(num / den - gaussDelta)
                bounds.append(num / den + gaussDelta)
        else:
            raise('Error: Incorrect interval type {}. Supported options are gaussian'.format(type))

        if not bounds:
            bounds = [0, 0]
        return bounds
