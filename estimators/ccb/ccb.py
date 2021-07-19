from estimators.ccb import base

class Estimator(base.Estimator):
    def __init__(self, FirstSlotEstimator):
        self.estimator = FirstSlotEstimator

    def add_example(self, p_logs, r, p_preds, count=1):
        """Expects lists for logged probabilities, rewards and predicted probabilities. These should correspond to each slot."""

        if not isinstance(p_logs, list) and not isinstance(r, list) and not isinstance(p_preds, list):
            raise('Error: p_logs, r and p_preds must be lists')

        if(len(p_logs) != len(p_preds) and len(p_logs) != len(r) and len(r) != len(p_preds)):
            raise('Error: p_logs, r and p_preds must be the same length, found {}, {} and {} respectively'.format(len(p_logs), len(r), len(p_preds)))

        self.estimator.add_example(p_logs[0], r[0], p_preds[0])

    def get(self):

        return self.estimator.get()

class Interval(base.Estimator):
    def __init__(self, FirstSlotInterval):
        interval = FirstSlotInterval

    def add_example(self, p_logs, r, p_preds, count=1):
        """Expects lists for logged probabilities, rewards and predicted probabilities. These should correspond to each slot."""

        if not isinstance(p_logs, list) and not isinstance(r, list) and not isinstance(p_preds, list):
            raise('Error: p_logs, r and p_preds must be lists')

        if(len(p_logs) != len(p_preds) and len(p_logs) != len(r) and len(r) != len(p_preds)):
            raise('Error: p_logs, r and p_preds must be the same length, found {}, {} and {} respectively'.format(len(p_logs), len(r), len(p_preds)))

        self.interval.add_example(p_logs[0], r[0], p_preds[0])

    def get(self):

        return self.interval.get()
