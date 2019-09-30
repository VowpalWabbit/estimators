# Empirical likehood estimator

import numpy as np

class Estimator:
    # NB: This works better you use the true wmin and wmax
    def __init__(self, wmin=0, wmax=np.inf):
        assert wmin < 1
        assert wmax > 1

        self.data = []
        self.n = 0
        self.wmin = wmin
        self.wmax = wmax

    def add_example(self, p_log, r, p_pred, count=1):
        if count > 0:
            self.data.append((count, p_pred/p_log, r))
            self.n += count
            self.wmax = max(self.wmax, p_pred/p_log)
            self.wmin = min(self.wmin, p_pred/p_log)

    def graddualobjective(self, beta):
       return sum(c * (w - 1)/((w - 1) * beta + self.n)
                  for c, w, _ in self.data)

    def get_estimate(self, rmin=0, rmax=1):
        from scipy.optimize import brentq

        assert self.n > 0, 'Error: No data point added'

        betaub = self.n / (1 - self.wmin)
        betamax = min(betaub,
                      min(( (self.n - c) / (1 - w)
                            for c, w, _ in self.data
                            if w < 1
                          ),
                          default=betaub))

        betalb = 0 if self.wmax == np.inf else self.n / (1 - self.wmax)
        betamin = max(betalb,
                      max(( (self.n - c) / (1 - w)
                            for c, w, _ in self.data
                            if w > 1
                          ),
                          default=betalb))

        gradmin = self.graddualobjective(betamin)
        gradmax = self.graddualobjective(betamax)

        if gradmin * gradmax < 0:
            betastar = brentq(f=self.graddualobjective, a=betamin, b=betamax)
        elif gradmin < 0:
            betastar = betamin
        else:
            betastar = betamax

        sumofw = sum(c * w / ((w - 1) * betastar + self.n)
                     for c, w, _ in self.data)
        remw = max(0.0, 1.0 - sumofw)

        vhat = sum(c * w * r / ((w - 1) * betastar + self.n)
                   for c, w, r in self.data)
        rhatmissing = sum(r for _, _, r in self.data) / self.n
        vhat += remw * rhatmissing

        return vhat
