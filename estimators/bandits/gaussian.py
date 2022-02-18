import math
from estimators.bandits import base
from scipy import stats
from typing import List


class Interval(base.Interval):
    def __init__(self):
        ################################# Aggregates quantities #########################################
        #
        # 'n':   IPS of numerator
        # 'N':   total number of samples in bin from log (IPS = n/N)
        # 'SoS': sum of squares of numerator's items (needed for Gaussian confidence intervals)
        #
        #################################################################################################

        self.data = {'n':0.,'N':0,'SoS':0}

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        self.data['N'] += count
        if p_pred > 0:
            p_over_p = p_pred/p_log
            if r != 0:
                self.data['n'] += r*p_over_p*count
                self.data['SoS'] += ((r*p_over_p)**2)*count

    def get(self, alpha: float = 0.05) -> List[float]:
        bounds = []
        num = self.data['n']
        den = self.data['N']
        sum_of_sq = self.data['SoS']

        if sum_of_sq > 0.0 and den > 1:
            z_gaussian_cdf = stats.norm.ppf(1-alpha/2)

            variance = (sum_of_sq - num * num / den) / (den - 1)
            gauss_delta = z_gaussian_cdf * math.sqrt(variance/den)
            bounds.append(num / den - gauss_delta)
            bounds.append(num / den + gauss_delta)

        if not bounds:
            bounds = [0, 0]
        return bounds
