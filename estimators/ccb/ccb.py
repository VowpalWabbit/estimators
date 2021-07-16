from estimators.ccb import base
from estimators.bandits import ips, mle, snips, cressieread

class Estimator(base.Estimator):
    def __init__(self):
        ips_estimator = ips.Estimator()
        snips_estimator = snips.Estimator()
        mle_estimator = mle.Estimator()
        cressieread_estimator = cressieread.Estimator()

    def add_example(self, p_logs, r, p_preds, count=1):
        """Expects lists for logged probabilities, rewards and predicted probabilities. These should correspond to each slot."""

        if not isinstance(p_logs, list) and not isinstance(r, list) and not isinstance(p_preds, list):
            raise('Error: p_logs, r and p_preds must be lists')

        if(len(p_logs) != len(p_preds) and len(p_logs) != len(r) and len(r) != len(p_preds)):
            raise('Error: p_logs, r and p_preds must be the same length, found {}, {} and {} respectively'.format(len(p_logs), len(r), len(p_preds)))

        elif type == 'ips':
            ips_estimator.add_example(p_logs[0], r[0], p_preds[0])

        elif type == 'snips':
            snips_estimator.add_example(p_logs[0], r[0], p_preds[0])

        elif type == 'mle':
            mle_estimator.add_example(p_logs[0], r[0], p_preds[0])

        elif type == 'cressieread':
            cressieread_estimator.add_example(p_logs[0], r[0], p_preds[0])

    def get(self, type):

        if type == 'ips':
            return ips_estimator.get()

        elif type == 'snips':
            return snips_estimator.get()

        elif type == 'mle':
            return mle_estimator.get()

        elif type == 'cressieread':
            return cressieread_estimator.get()
