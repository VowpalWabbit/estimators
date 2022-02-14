# CR(-2) is particularly computationally convenient

from math import fsum, inf
from estimators.bandits import base
from typing import List

class Estimator(base.Estimator):
    # NB: This works better you use the true wmin and wmax
    #     which is _not_ the empirical minimum and maximum
    #     but rather the actual smallest and largest possible values
    def __init__(self, wmin: float = 0, wmax: float = inf):
        assert wmin < 1
        assert wmax > 1

        self.wmin = wmin
        self.wmax = wmax

        self.data = []

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        if count > 0:
            w = p_pred / p_log
            assert w >= 0, 'Error: negative importance weight'

            self.data.append((count, w, r))
            self.wmax = max(self.wmax, w)
            self.wmin = min(self.wmin, w)

    def get(self) -> float:
        n = fsum(c for c, _, _ in self.data)
        assert n > 0, 'Error: No data point added'

        sumw = fsum(c * w for c, w, _ in self.data)
        sumwsq = fsum(c * w**2 for c, w, _ in self.data)
        sumwr = fsum(c * w * r for c, w, r in self.data)
        sumwsqr = fsum(c * w**2 * r for c, w, r in self.data)
        sumr = fsum(c * r for c, _, r in self.data)

        wfake = self.wmax if sumw < n else self.wmin

        if wfake == inf:
            gamma = -(1 + n) / n
            beta = 0
        else:
            a = (wfake + sumw) / (1 + n)
            b = (wfake**2 + sumwsq) / (1 + n)
            assert a*a < b
            gamma = (b - a) / (a*a - b)
            beta = (1 - a) / (a*a - b)

        vhat = (-gamma * sumwr - beta * sumwsqr) / (1 + n)
        missing = max(0, 1 - (-gamma * sumw - beta * sumwsq) / (1 + n))
        rhatmissing = sumr / n
        vhat += missing * rhatmissing

        return vhat

class Interval(base.Interval):
    # NB: This works better you use the true wmin and wmax
    #     which is _not_ the empirical minimum and maximum
    #     but rather the actual smallest and largest possible values
    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1):
        assert wmin < 1
        assert wmax > 1

        self.wmin = wmin
        self.wmax = wmax

        self.rmin = rmin
        self.rmax = rmax

        self.data = []

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        if count > 0:
            w = p_pred / p_log
            assert w >= 0, 'Error: negative importance weight'

            self.data.append((count, w, r))
            self.wmax = max(self.wmax, w)
            self.wmin = min(self.wmin, w)

    def get(self, alpha: float = 0.05) -> List[float]:
        from math import isclose, sqrt
        from scipy.stats import f

        n = fsum(c for c, _, _ in self.data)
        assert n > 0, 'Error: No data point added'

        sumw = fsum(c * w for c, w, _ in self.data)
        sumwsq = fsum(c * w**2 for c, w, _ in self.data)
        sumwr = fsum(c * w * r for c, w, r in self.data)
        sumwsqr = fsum(c * w**2 * r for c, w, r in self.data)
        sumwsqrsq = fsum(c * w**2 * r**2 for c, w, r in self.data)

        uncwfake = self.wmax if sumw < n else self.wmin
        if uncwfake == inf:
           uncgstar = 1 + 1 / n
        else:
           unca = (uncwfake + sumw) / (1 + n)
           uncb = (uncwfake**2 + sumwsq) / (1 + n)
           uncgstar = (1 + n) * (unca - 1)**2 / (uncb - unca*unca)

        Delta = f.isf(q=alpha, dfn=1, dfd=n)
        phi = (-uncgstar - Delta) / (2 * (1 + n))

        bounds = []
        for r, sign in ((self.rmin, 1), (self.rmax, -1)):
            candidates = []
            for wfake in (self.wmin, self.wmax):
                if wfake == inf:
                    x = sign * (r + (sumwr - sumw * r) / n)
                    y = (  (r * sumw - sumwr)**2 / (n * (1 + n))
                         - (r**2 * sumwsq - 2 * r * sumwsqr + sumwsqrsq) / (1 + n)
                        )
                    z = phi + 1 / (2 * n)
                    if isclose(y*z, 0, abs_tol=1e-9):
                        y = 0

                    if z <= 0 and y * z >= 0:
                        kappa = sqrt(y / (2 * z))
                        if isclose(kappa, 0):
                            candidates.append(sign * r)
                        else:
                            gstar = x - sqrt(2 * y * z)

                            candidates.append(gstar)
                else:
                    barw = (wfake + sumw) / (1 + n)
                    barwsq = (wfake*wfake + sumwsq) / (1 + n)
                    barwr = sign * (wfake * r + sumwr) / (1 + n)
                    barwsqr = sign * (wfake * wfake * r + sumwsqr) / (1 + n)
                    barwsqrsq = (wfake * wfake * r * r + sumwsqrsq) / (1 + n)

                    if barwsq > barw**2:
                        x = barwr + ((1 - barw) * (barwsqr - barw * barwr) / (barwsq - barw**2))
                        y = (barwsqr - barw * barwr)**2 / (barwsq - barw**2) - (barwsqrsq - barwr**2)
                        z = phi + (1/2) * (1 - barw)**2 / (barwsq - barw**2)

                        if isclose(y*z, 0, abs_tol=1e-9):
                            y = 0

                        if z <= 0 and y * z >= 0:
                            kappa = sqrt(y / (2 * z)) if y * z > 0 else 0
                            if isclose(kappa, 0):
                                candidates.append(sign * r)
                            else:
                                gstar = x - sqrt(2 * y * z)
                                candidates.append(gstar)

            best = min(candidates)
            vbound = min(self.rmax, max(self.rmin, sign*best))
            bounds.append(vbound)

        return bounds
