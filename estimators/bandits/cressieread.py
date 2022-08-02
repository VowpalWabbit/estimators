# CR(-2) is particularly computationally convenient

from math import inf
from estimators.bandits import base
from typing import List, Optional
from estimators.math import IncrementalFsum


class EstimatorImpl:
    wmin: float
    wmax: float
    n: IncrementalFsum
    sumw: IncrementalFsum
    sumwsq: IncrementalFsum
    sumwr: IncrementalFsum
    sumwsqr: IncrementalFsum
    sumr: IncrementalFsum

    # NB: This works better you use the true wmin and wmax
    #     which is _not_ the empirical minimum and maximum
    #     but rather the actual smallest and largest possible values

    def __init__(self, wmin: float = 0, wmax: float = inf):
        assert wmin < 1
        assert wmax > 1

        self.wmin = wmin
        self.wmax = wmax

        self.n = IncrementalFsum()
        self.sumw = IncrementalFsum()
        self.sumwsq = IncrementalFsum()
        self.sumwr = IncrementalFsum()
        self.sumwsqr = IncrementalFsum()
        self.sumr = IncrementalFsum()

    def add(self, w: float, r: float, count: float = 1.0) -> None:
        if count > 0:
            assert w >= 0, 'Error: negative importance weight'

            self.n += count
            self.sumw += count * w
            self.sumwsq += count * w ** 2
            self.sumwr += count * w * r
            self.sumwsqr += count * w ** 2 * r
            self.sumr += count * r

            self.wmax = max(self.wmax, w)
            self.wmin = min(self.wmin, w)

    def get(self) -> Optional[float]:
        n = float(self.n)
        if n == 0:
            return None

        sumw = float(self.sumw)
        sumwsq = float(self.sumwsq)
        sumwr = float(self.sumwr)
        sumwsqr = float(self.sumwsqr)
        sumr = float(self.sumr)

        wfake = self.wmax if sumw < n else self.wmin

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

        return vhat

    def __add__(self, other: 'EstimatorImpl') -> 'EstimatorImpl':
        result = EstimatorImpl(
            wmin=min(self.wmin, other.wmin),
            wmax=max(self.wmax, other.wmax))

        result.n = self.n + other.n
        result.sumw = self.sumw + other.sumw
        result.sumwsq = self.sumwsq + other.sumwsq
        result.sumwr = self.sumwr + other.sumwr
        result.sumwsqr = self.sumwsqr + other.sumwsqr
        result.sumr = self.sumr + other.sumr

        return result


class IntervalImpl:
    wmin: float
    wmax: float
    rmin: float
    rmax: float
    n: IncrementalFsum
    sumw: IncrementalFsum
    sumwsq: IncrementalFsum
    sumwr: IncrementalFsum
    sumwsqr: IncrementalFsum
    sumwsqrsq: IncrementalFsum
    empirical_r_bounds: bool

    # NB: This works better you use the true wmin and wmax
    #     which is _not_ the empirical minimum and maximum
    #     but rather the actual smallest and largest possible values

    def __init__(self, wmin: float, wmax: float, rmin: float, rmax: float, empirical_r_bounds: bool):
        assert wmin < 1
        assert wmax > 1

        self.wmin = wmin
        self.wmax = wmax
        self.rmin = rmin
        self.rmax = rmax
        self.empirical_r_bounds = empirical_r_bounds

        self.n = IncrementalFsum()
        self.sumw = IncrementalFsum()
        self.sumwsq = IncrementalFsum()
        self.sumwr = IncrementalFsum()
        self.sumwsqr = IncrementalFsum()
        self.sumwsqrsq = IncrementalFsum()

    def add(self, w: float, r: float, count: float = 1.0) -> None:
        assert count == 1.0, "need to explicitly model the pdrop generatively in order to prevent misleading confidence interval widths"
        if count > 0:
            assert w >= 0, 'Error: negative importance weight'

            self.n += count
            self.sumw += count * w
            self.sumwsq += count * w ** 2
            self.sumwr += count * w * r
            self.sumwsqr += count * w ** 2 * r
            self.sumwsqrsq += count * w ** 2 * r ** 2

            self.wmax = max(self.wmax, w)
            self.wmin = min(self.wmin, w)

            if self.empirical_r_bounds:
                self.rmax = max(self.rmax, r)
                self.rmin = min(self.rmin, r)
            else:
                if r > self.rmax or r < self.rmin:
                    raise ValueError(f'Error: Value of r={r} is outside rmin={self.rmin}, rmax={self.rmax} bounds')

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> List[Optional[float]]:
        from math import isclose, sqrt
        from scipy.stats import f

        n = float(self.n)
        if n == 0:
            return [None, None]

        sumw = float(self.sumw)
        sumwsq = float(self.sumwsq)
        sumwr = float(self.sumwr)
        sumwsqr = float(self.sumwsqr)
        sumwsqrsq = float(self.sumwsqrsq)

        uncwfake = self.wmax if sumw < n else self.wmin
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
            for wfake in (self.wmin, self.wmax):
                if wfake == inf:
                    x = sign * (r + (sumwr - sumw * r) / n)
                    y = ((r * sumw - sumwr) ** 2 / (n * (1 + n))
                         - (r ** 2 * sumwsq - 2 * r * sumwsqr + sumwsqrsq) / (1 + n)
                         )
                    z = phi + 1 / (2 * n)
                    if isclose(y * z, 0, abs_tol=1e-9):
                        y = 0

                    if isclose(y * z, 0, abs_tol=atol * atol):
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

                        if isclose(y * z, 0, abs_tol=atol * atol):
                            gstar = x - sqrt(2) * atol
                            candidates.append(gstar)
                        elif z <= 0 and y * z >= 0:
                            gstar = x - sqrt(2 * y * z)
                            candidates.append(gstar)

            best = min(candidates)
            vbound = min(self.rmax, max(self.rmin, sign * best))
            bounds.append(vbound)

        return bounds

    def __add__(self, other: 'IntervalImpl') -> 'IntervalImpl':
        assert not (self.empirical_r_bounds ^ other.empirical_r_bounds), 'Summation of estimators with various r bounds policy is prohibited'
        
        if not self.empirical_r_bounds:
            assert self.rmin == other.rmin, 'Summation of estimators with various r bounds is prohibited'
            assert self.rmax == other.rmax, 'Summation of estimators with various r bounds is prohibited'

        result = IntervalImpl(
            wmin=min(self.wmin, other.wmin),
            wmax=max(self.wmax, other.wmax),
            rmin=min(self.rmin, other.rmin),
            rmax=max(self.rmax, other.rmax),
            empirical_r_bounds=self.empirical_r_bounds 
        )

        result.n = self.n + other.n
        result.sumw = self.sumw + other.sumw
        result.sumwsq = self.sumwsq + other.sumwsq
        result.sumwr = self.sumwr + other.sumwr
        result.sumwsqr = self.sumwsqr + other.sumwsqr
        result.sumwsqrsq = self.sumwsqrsq + other.sumwsqrsq

        return result


class Estimator(base.Estimator):
    _impl: EstimatorImpl

    def __init__(self, wmin: float = 0, wmax: float = inf):
        self._impl = EstimatorImpl(wmin, wmax)

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        self._impl.add(p_pred / p_log, r, count)

    def get(self) -> Optional[float]:
        return self._impl.get()

    def __add__(self, other: 'Estimator') -> 'Estimator':
        result = Estimator()
        result._impl = self._impl + other._impl
        return result


class Interval(base.Interval):
    _impl: IntervalImpl

    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1, empirical_r_bounds = False):
        self._impl = IntervalImpl(wmin, wmax, rmin, rmax, empirical_r_bounds)

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        self._impl.add(p_pred / p_log, r, count)

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> List[Optional[float]]:
        return self._impl.get(alpha, atol)

    def __add__(self, other: 'Interval') -> 'Interval':
        result = Interval()
        result._impl = self._impl + other._impl
        return result
