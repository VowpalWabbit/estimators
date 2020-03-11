import math
from sklearn.linear_model import LinearRegression
import numpy as np
import pudb


class Estimator:
    def __init__(self):
        ############################### Aggregates quantities ######################################
        #
        # 'pred':   probabilities
        # 'N':   total number of samples
        # 'r':   costs
        #
        #################################################################################################

        self.data = {'pred': [], 'r': [], 'N': 0}

    def add_example(self, p_log, r, p_pred, count=1):
        self.data['N'] += count
        if count > 1:
            self.data['pred'].extend(p_pred)
            self.data['r'].extend(r)
        elif count == 1:
            self.data['pred'].append(p_pred)
            self.data['r'].append(r)
        
    def get_estimate(self):
        if self.data['N'] == 0:
            raise('Error: No data point added')

        reg = LinearRegression().fit(np.array(self.data['r']).reshape(-1, 1), np.array(self.data['pred']).reshape(-1, 1))
        return reg.coef_[0][0], reg.intercept_[0]
