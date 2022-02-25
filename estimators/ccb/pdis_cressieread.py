from estimators.ccb import base
from math import fsum, inf
from typing import List
from estimators.math import IncrementalFsum


def _get_safe(values, index, default):
    return values[index] if index < len(values) else default


class _Impl:
    wmin: float
    wmax: float
    maxstep: int
    n: IncrementalFsum
    sumw: List[IncrementalFsum]
    sumwsq: List[IncrementalFsum]
    sumwr: List[IncrementalFsum]
    sumwsqr: List[IncrementalFsum]  
    sumr: List[IncrementalFsum]

    def __init__(self, wmin: float = 0, wmax: float = inf):
        assert wmin < 1
        assert wmax > 1

        self.wmin = wmin
        self.wmax = wmax
        self.maxstep = 0

        self.n = IncrementalFsum()
        self.sumw = []
        self.sumwsq = []
        self.sumwr = []
        self.sumwsqr = []
        self.sumr = []

    def _resize_statistics(self, statistics, constructor):
        constructor = IncrementalFsum if len(statistics) == 0 else constructor
        for i in range(max(self.maxstep - len(statistics), 0)):
            statistics.append(constructor())

    def add(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        from copy import deepcopy
        if count > 0:
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            assert all(w >= 0 for w in ws), 'Error: negative importance weight'
            self.maxstep = max(self.maxstep, len(ws))
            self._resize_statistics(self.sumw, lambda: deepcopy(self.sumw[-1]))
            self._resize_statistics(self.sumwsq, lambda: deepcopy(self.sumwsq[-1]))
            self._resize_statistics(self.sumwr, IncrementalFsum)
            self._resize_statistics(self.sumwsqr, IncrementalFsum)
            self._resize_statistics(self.sumr, IncrementalFsum)
            w = 1.0
            self.n += count
            for i in range(self.maxstep):
                w *= _get_safe(ws, i, 1)
                self.sumw[i] += count * w
                self.sumwsq[i] += count * w ** 2
                self.sumwr[i] += count * w * _get_safe(rs, i, 0)
                self.sumwsqr[i] += count * w ** 2 * _get_safe(rs, i, 0)
                self.sumr[i] += count * _get_safe(rs, i, 0)

            self.wmax = max(self.wmax, max(ws))
            self.wmin = min(self.wmin, min(ws))


class Estimator(base.Estimator):
    _impl: _Impl

    def __init__(self, wmin: float = 0, wmax: float = inf):
        self._impl = _Impl(wmin, wmax)


    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        self._impl.add(p_logs, rs, p_preds, count)


    def get(self) -> List[float]:
        n = float(self._impl.n)
        if n == 0:
            return []

        stepvhats = []
        for step in range(self._impl.maxstep):
            sumw = float(self._impl.sumw[step])
            sumwsq = float(self._impl.sumwsq[step])
            sumwr = float(self._impl.sumwr[step])
            sumwsqr = float(self._impl.sumwsqr[step])
            sumr = float(self._impl.sumr[step])

            wfake = self._impl.wmax ** (step + 1) if sumw < n else self._impl.wmin ** (step + 1)

            if wfake == inf:
                gamma = -(1 + n) / n
                beta = 0
            else:
                a = (wfake + sumw) / (1 + n)
                b = (wfake ** 2 + sumwsq) / (1 + n)
                assert a * a < b
                gamma = (b - a) / (a * a - b)
                beta = (1 - a) / (a * a - b)

            vhat = (-gamma * sumwr - beta * sumwsqr) / (1 + n)
            missing = max(0.0, 1 - (-gamma * sumw - beta * sumwsq) / (1 + n))
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
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            assert all(w >= 0 for w in ws), 'Error: negative importance weight'

            self.data.append((count, ws, rs))
            self.wmax = max(self.wmax, max(ws))
            self.wmin = min(self.wmin, min(ws))
            self.maxstep = max(self.maxstep, len(ws))

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> List[List[float]]:
        from math import isclose, sqrt
        from scipy.stats import f

        def prod(vs):
            import operator
            from functools import reduce
            return reduce(operator.mul, vs, 1)

        n = fsum(c for c, _, _ in self.data)
        if n == 0:
            return []

        stepbounds = []

        for step in range(1, self.maxstep + 1):
            sumw = fsum(c * w for c, ws, _ in self.data
                        for w in (prod(ws[:min(step, len(ws))]),))
            sumwsq = fsum(c * w ** 2 for c, ws, _ in self.data
                          for w in (prod(ws[:min(step, len(ws))]),))
            sumwr = fsum(c * w * _get_safe(rs, step - 1, 0) for c, ws, rs in self.data
                         for w in (prod(ws[:min(step, len(ws))]),))
            sumwsqr = fsum(c * w ** 2 * _get_safe(rs, step - 1, 0) for c, ws, rs in self.data
                           for w in (prod(ws[:min(step, len(ws))]),))
            sumwsqrsq = fsum(c * w ** 2 * _get_safe(rs, step - 1, 0) ** 2 for c, ws, rs in self.data
                             for w in (prod(ws[:min(step, len(ws))]),))

            uncwfake = self.wmax ** step if sumw < n else self.wmin ** step
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
