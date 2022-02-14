from estimators.ccb import base
from math import fsum, inf
from typing import List

class Estimator(base.Estimator):
    def __init__(self, wmin: float = 0, wmax: float = inf):
        assert wmin < 1
        assert wmax > 1

        self.wmin = wmin
        self.wmax = wmax
        self.maxstep = 0

        self.data = []

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        if count > 0:
            ws = [ p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs) ]
            assert all(w >= 0 for w in ws), 'Error: negative importance weight'

            self.data.append((count, ws, rs))
            self.wmax = max(self.wmax, max(ws))
            self.wmin = min(self.wmin, min(ws))
            self.maxstep = max(self.maxstep, len(ws))

    def get(self) -> List[float]:
        def prod(vs):
            import operator
            try:
                return reduce(operator.mul, vs, 1)
            except:
                from functools import reduce
                return reduce(operator.mul, vs, 1)

        n = fsum(c for c, _, _ in self.data)
        assert n > 0, 'Error: No data point added'

        stepvhats = []
        for step in range(self.maxstep):
            sumw = fsum(c * w for c, ws, _ in self.data
                        for w in (prod(ws[:step]),))
            sumwsq = fsum(c * w**2 for c, ws, _ in self.data
                          for w in (prod(ws[:step]),))
            sumwr = fsum(c * w * rs[step] for c, ws, rs in self.data
                         for w in (prod(ws[:step]),))
            sumwsqr = fsum(c * w**2 * rs[step] for c, ws, rs in self.data
                           for w in (prod(ws[:step]),))
            sumr = fsum(c * rs[step] for c, _, rs in self.data)
            wfake = self.wmax**(1+step) if sumw < n else self.wmin**(1+step)

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

            stepvhats.append(vhat)

        return stepvhats

class Interval(base.Interval):
    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1):
        assert wmin < 1
        assert wmax > 1

        self.wmin = wmin
        self.wmax = wmax
        self.maxstep = 0

        self.rmin = rmin
        self.rmax = rmax

        self.data = []

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        if count > 0:
            ws = [ p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs) ]
            assert all(w >= 0 for w in ws), 'Error: negative importance weight'

            self.data.append((count, ws, rs))
            self.wmax = max(self.wmax, max(ws))
            self.wmin = min(self.wmin, min(ws))
            self.maxstep = max(self.maxstep, len(ws))

    def get(self, alpha: float = 0.05) -> List[List[float]]:
        from math import isclose, sqrt
        from scipy.stats import f

        def prod(vs):
            import operator
            try:
                return reduce(operator.mul, vs, 1)
            except:
                from functools import reduce
                return reduce(operator.mul, vs, 1)

        n = fsum(c for c, _, _ in self.data)
        assert n > 0, 'Error: No data point added'

        stepbounds = []

        for step in range(self.maxstep):
            sumw = fsum(c * w for c, ws, _ in self.data
                              for w in (prod(ws[:step]),))
            sumwsq = fsum(c * w**2 for c, ws, _ in self.data
                              for w in (prod(ws[:step]),))
            sumwr = fsum(c * w * rs[step] for c, ws, rs in self.data
                                           for w in (prod(ws[:step]),))
            sumwsqr = fsum(c * w**2 * rs[step] for c, ws, rs in self.data
                                                for w in (prod(ws[:step]),))
            sumwsqrsq = fsum(c * w**2 * rs[step]**2 for c, ws, rs in self.data
                                                    for w in (prod(ws[:step]),))

            uncwfake = self.wmax**(1+step) if sumw < n else self.wmin**(1+step)
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
                for wfake in (self.wmin**(1+step), self.wmax**(1+step)):
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

            stepbounds.append(bounds)

        return stepbounds