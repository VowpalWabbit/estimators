from estimators.bandits import base


class Estimator(base.Estimator):
    def __init__(self):
        ################################# Aggregates quantities #########################################
        #
        # 'n':   IPS of numerator
        # 'N':   total number of samples in bin from log (IPS = n/N)
        #
        #################################################################################################

        self.data = {'n':0.,'N':0}

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        self.data['N'] += count
        if p_pred > 0:
            p_over_p = p_pred/p_log
            if r != 0:
                self.data['n'] += r*p_over_p*count

    def get(self) -> float:
        if self.data['N'] == 0:
            raise ValueError('Error: No data point added')

        return self.data['n']/self.data['N']
