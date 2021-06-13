import math
from scipy.stats import beta
from cb_base import Estimator

class SNIPSestimator(Estimator):

    def __init__(self):
        ################################# Aggregates quantities #########################################
        #
        # 'n':   IPS of numerator
        # 'N':   total number of samples in bin from log (IPS = n/N)
        # 'd':   IPS of denominator (SNIPS = n/d)
        #
        #################################################################################################

        self.data = {'n':0.,'N':0,'d':0.,'c':0.,'SoS':0}

    def add_example(self, p_log, r, p_pred, count=1):
        self.data['N'] += count
        if p_pred > 0:
            p_over_p = p_pred/p_log
            self.data['d'] += p_over_p*count
            if r != 0:
                self.data['n'] += r*p_over_p*count

    def get_estimate(self):
        if self.data['N'] == 0:
            raise('Error: No data point added')

        if self.data['d'] != 0:
            return self.data['n']/self.data['d']
        else:
            return 0
