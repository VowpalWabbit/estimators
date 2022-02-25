from estimators.ccb import base
from math import fsum, inf
from typing import List, Callable
from estimators.math import IncrementalFsum
from estimators.bandits.cressieread import EstimatorImpl


def _get_safe(values, index, default):
    return values[index] if index < len(values) else default


def _resize_statistics(statistics, count, constructor):
    for _ in range(max(count - len(statistics), 0)):
        statistics.append(constructor())

 
class Estimator(base.Estimator):
    _impl_ctr: Callable[[int], EstimatorImpl]
    _impl: List[EstimatorImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf):
        self._impl_ctr = lambda step: EstimatorImpl(wmin, wmax, step)
        self._impl = [self._impl_ctr(0)]


    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        if count > 0:
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            w = 1.0
            for i in range(len(ws)):
                w *= ws[i]
                if len(self._impl) <= i:
                    self._impl.append(self._impl_ctr(i))
                self._impl[i].add(w, rs[i], count)


    def get(self) -> List[float]:
        result = []
        n0 = float(self._impl[0].n)
        if n0 > 0:
            for impl in self._impl:
                result.append(impl.get() * float(impl.n) / n0 )
        return result


class Interval(base.Interval):
    wmin: float
    wmax: float
    maxstep: int
    rmin: float
    rmax: float
    n: IncrementalFsum
    sumw: List[IncrementalFsum]
    sumwsq: List[IncrementalFsum]
    sumwr: List[IncrementalFsum]
    sumwsqr: List[IncrementalFsum]
    sumwsqrsq: List[IncrementalFsum]

    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1):
        assert wmin < 1
        assert wmax > 1

        self.wmin = wmin
        self.wmax = wmax
        self.maxstep = 0

        self.rmin = rmin
        self.rmax = rmax

        self.n = IncrementalFsum()
        self.sumw = [IncrementalFsum()]
        self.sumwsq = [IncrementalFsum()]
        self.sumwr = [IncrementalFsum()]
        self.sumwsqr = [IncrementalFsum()]
        self.sumwsqrsq = [IncrementalFsum()]

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        from copy import deepcopy
        if count > 0:
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            assert all(w >= 0 for w in ws), 'Error: negative importance weight'
            self.maxstep = max(self.maxstep, len(ws))
            _resize_statistics(self.sumw, self.maxstep, lambda: deepcopy(self.sumw[-1]))
            _resize_statistics(self.sumwsq, self.maxstep, lambda: deepcopy(self.sumwsq[-1]))
            _resize_statistics(self.sumwr, self.maxstep, IncrementalFsum)
            _resize_statistics(self.sumwsqr, self.maxstep, IncrementalFsum)
            _resize_statistics(self.sumwsqrsq, self.maxstep, IncrementalFsum)
            w = 1.0
            self.n += count
            for i in range(self.maxstep):
                w *= _get_safe(ws, i, 1)
                r = _get_safe(rs, i, 0)
                self.sumw[i] += count * w
                self.sumwsq[i] += count * w ** 2
                self.sumwr[i] += count * w * r
                self.sumwsqr[i] += count * w ** 2 * r
                self.sumwsqrsq[i] += count * w ** 2 * r ** 2

            self.wmax = max(self.wmax, max(ws))
            self.wmin = min(self.wmin, min(ws))

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> List[List[float]]:
        from math import isclose, sqrt
        from scipy.stats import f

        n = float(self.n)
        if n == 0:
            return []

        stepbounds = []

        for step in range(self.maxstep):
            sumw = float(self.sumw[step])
            sumwsq = float(self.sumwsq[step])
            sumwr = float(self.sumwr[step])
            sumwsqr = float(self.sumwsqr[step])
            sumwsqrsq = float(self.sumwsqrsq[step])

            uncwfake = self.wmax ** (step + 1) if sumw < n else self.wmin ** (step + 1)
            if uncwfake == inf:
                uncgstar = 1 + 1 / n
            else:
                unca = (uncwfake + sumw) / (1 + n)
                uncb = (uncwfake ** 2 + sumwsq) / (1 + n)
                uncgstar = (1 + n) * (unca - 1) ** 2 / (uncb - unca * unca)

            delta = f.isf(q=alpha, dfn=1, dfd=n)
            phi = (-uncgstar - delta) / (2 * (1 + n))

            bounds = []
            for r, sign in ((self.rmin, 1), (self.rmax, -1)):
                candidates = []
                for wfake in (self.wmin ** (1 + step), self.wmax ** (1 + step)):
                    if wfake == inf:
                        x = sign * (r + (sumwr - sumw * r) / n)
                        y = ((r * sumw - sumwr) ** 2 / (n * (1 + n))
                             - (r ** 2 * sumwsq - 2 * r * sumwsqr + sumwsqrsq) / (1 + n)
                             )
                        z = phi + 1 / (2 * n)
                        if isclose(y * z, 0, abs_tol=atol):
                            gstar = x - sqrt(2) * atol
                            candidates.append(gstar)
                        elif z <= 0 and y * z >= 0:
                            gstar = x - sqrt(2 * y * z)
                            candidates.append(gstar)
                    else:
                        barw = (wfake + sumw) / (1 + n)
                        barwsq = (wfake * wfake + sumwsq) / (1 + n)
                        barwr = sign * (wfake * r + sumwr) / (1 + n)
                        barwsqr = sign * (wfake * wfake * r + sumwsqr) / (1 + n)
                        barwsqrsq = (wfake * wfake * r * r + sumwsqrsq) / (1 + n)

                        if barwsq > barw ** 2:
                            x = barwr + ((1 - barw) * (barwsqr - barw * barwr) / (barwsq - barw ** 2))
                            y = (barwsqr - barw * barwr) ** 2 / (barwsq - barw ** 2) - (barwsqrsq - barwr ** 2)
                            z = phi + (1 / 2) * (1 - barw) ** 2 / (barwsq - barw ** 2)

                            if isclose(y * z, 0, abs_tol=atol):
                                gstar = x - sqrt(2) * atol
                                candidates.append(gstar)
                            elif z <= 0 and y * z >= 0:
                                gstar = x - sqrt(2 * y * z)
                                candidates.append(gstar)

                best = min(candidates)
                vbound = min(self.rmax, max(self.rmin, sign * best))
                bounds.append(vbound)

            stepbounds.append(bounds)

        return stepbounds
