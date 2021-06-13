import math
from scipy.stats import beta
from cb_base import Estimator
from cb_base import Interval

class IPSestimator(Estimator):

    def __init__(self):
        ################################# Aggregates quantities #########################################
        #
        # 'n':   IPS of numerator
        # 'N':   total number of samples in bin from log (IPS = n/N)
        #
        #################################################################################################

        self.data = {'n':0.,'N':0}

    def add_example(self, p_log, r, p_pred, count=1):
        self.data['N'] += count
        if p_pred > 0:
            p_over_p = p_pred/p_log
            if r != 0:
                self.data['n'] += r*p_over_p*count

    def get_estimate(self):
        if self.data['N'] == 0:
            raise('Error: No data point added')

        return self.data['n']/self.data['N']

class IPSinterval(Interval):

    def __init__(self):
        ################################# Aggregates quantities #########################################
        #
        # 'n':   IPS of numerator
        # 'N':   total number of samples in bin from log (IPS = n/N)
        # 'c':   max abs. value of numerator's items (needed for Clopper-Pearson confidence intervals)
        # 'SoS': sum of squares of numerator's items (needed for Gaussian confidence intervals)
        #
        #################################################################################################

        self.data = {'n':0.,'N':0,'c':0.,'SoS':0}

    def add_example(self, p_log, r, p_pred, count=1):
        self.data['N'] += count
        if p_pred > 0:
            p_over_p = p_pred/p_log
            if r != 0:
                self.data['n'] += r*p_over_p*count
                self.data['c'] = max(self.data['c'], r*p_over_p)
                self.data['SoS'] += ((r*p_over_p)**2)*count

    def get_interval(self, type, alpha=0.05):
        bounds = []
        num = self.data['n']
        den = self.data['N']
        maxWeightedCost = self.data['c']
        SoS = self.data['SoS']

        if type == "clopper-pearson":
            if maxWeightedCost > 0.0:
                successes = num / maxWeightedCost
                n = den / maxWeightedCost
                bounds.append(beta.ppf(alpha / 2, successes, n - successes + 1))
                bounds.append(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
        elif type == "gaussian":
            if SoS > 0.0 and den > 1:
                zGaussianCdf = {
                  0.25: 1.15,
                  0.1: 1.645,
                  0.05: 1.96
                }

                variance = (SoS - num * num / den) / (den - 1)
                gaussDelta = zGaussianCdf[alpha] * math.sqrt(variance/den)
                bounds.append(num / den - gaussDelta)
                bounds.append(num / den + gaussDelta)

        if not bounds:
            bounds = [0, 0]
        return bounds