# CR(-2) is particularly computationally convenient

from __future__ import annotations

from math import inf
from estimators.bandits import base
from typing import List, Optional, Tuple
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

    def add(self, w: float, r: float) -> None:
        assert w >= 0, "Error: negative importance weight"

        self.n += 1
        self.sumw += w
        self.sumwsq += w**2
        self.sumwr += w * r
        self.sumwsqr += w**2 * r
        self.sumr += r

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
            beta = 0.0
        else:
            a = (wfake + sumw) / (1 + n)
            b = (wfake**2 + sumwsq) / (1 + n)
            assert a * a < b
            gamma = (b - a) / (a * a - b)
            beta = (1 - a) / (a * a - b)

        vhat = (-gamma * sumwr - beta * sumwsqr) / (1 + n)
        missing = max(0.0, 1 - (-gamma * sumw - beta * sumwsq) / (1 + n))
        rhatmissing = sumr / n
        vhat += missing * rhatmissing

        return vhat

    def __add__(self, other: "EstimatorImpl") -> "EstimatorImpl":
        result = EstimatorImpl(
            wmin=min(self.wmin, other.wmin), wmax=max(self.wmax, other.wmax)
        )

        result.n = IncrementalFsum.merge(self.n, other.n)
        result.sumw = IncrementalFsum.merge(self.sumw, other.sumw)
        result.sumwsq = IncrementalFsum.merge(self.sumwsq, other.sumwsq)
        result.sumwr = IncrementalFsum.merge(self.sumwr, other.sumwr)
        result.sumwsqr = IncrementalFsum.merge(self.sumwsqr, other.sumwsqr)
        result.sumr = IncrementalFsum.merge(self.sumr, other.sumr)

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

    def __init__(
        self,
        wmin: float,
        wmax: float,
        rmin: float,
        rmax: float,
        empirical_r_bounds: bool,
    ):
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

    def add(self, w: float, r: float, p_drop: float, n_drop: Optional[int]) -> None:
        assert w >= 0, "Error: negative importance weight"
        n_drop_tmp = float(n_drop) if n_drop is not None else p_drop / (1 - p_drop)
        w /= 1 - p_drop
        self.n += 1 + n_drop_tmp
        self.sumw += w
        self.sumwsq += w**2
        self.sumwr += w * r
        self.sumwsqr += w**2 * r
        self.sumwsqrsq += w**2 * r**2

        self.wmax = max(self.wmax, w)
        self.wmin = min(self.wmin, w)

        if self.empirical_r_bounds:
            self.rmax = max(self.rmax, r)
            self.rmin = min(self.rmin, r)
        else:
            if r > self.rmax or r < self.rmin:
                raise ValueError(
                    f"Error: Value of r={r} is outside rmin={self.rmin}, rmax={self.rmax} bounds"
                )

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> Tuple[float, float]:
        from math import isclose, sqrt
        from scipy.stats import f  # type: ignore

        n = float(self.n)
        if n == 0:
            return (-inf, inf) if self.empirical_r_bounds else (self.rmin, self.rmax)

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
            uncb = (uncwfake**2 + sumwsq) / (1 + n)
            uncgstar = (1 + n) * (unca - 1) ** 2 / (uncb - unca * unca)

        delta = f.isf(q=alpha, dfn=1, dfd=n)
        phi = (-uncgstar - delta) / (2 * (1 + n))

        bounds: List[float] = []
        for r, sign in ((self.rmin, 1), (self.rmax, -1)):
            candidates = []
            for wfake in (self.wmin, self.wmax):
                if wfake == inf:
                    x = sign * (r + (sumwr - sumw * r) / n)
                    y = (r * sumw - sumwr) ** 2 / (n * (1 + n)) - (
                        r**2 * sumwsq - 2 * r * sumwsqr + sumwsqrsq
                    ) / (1 + n)
                    z = phi + 1 / (2 * n)

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

                    if barwsq > barw**2:
                        x = barwr + (
                            (1 - barw) * (barwsqr - barw * barwr) / (barwsq - barw**2)
                        )
                        y = (barwsqr - barw * barwr) ** 2 / (barwsq - barw**2) - (
                            barwsqrsq - barwr**2
                        )
                        z = phi + (1 / 2) * (1 - barw) ** 2 / (barwsq - barw**2)

                        if isclose(y * z, 0, abs_tol=atol * atol):
                            gstar = x - sqrt(2) * atol
                            candidates.append(gstar)
                        elif z <= 0 and y * z >= 0:
                            gstar = x - sqrt(2 * y * z)
                            candidates.append(gstar)

            if candidates and len(candidates) > 0:
                best = min(candidates)
            else:
                best = self.rmin
            vbound = min(self.rmax, max(self.rmin, sign * best))
            bounds.append(vbound)

        return (bounds[0], bounds[1])

    def __add__(self, other: "IntervalImpl") -> "IntervalImpl":
        assert not (
            self.empirical_r_bounds ^ other.empirical_r_bounds
        ), "Summation of estimators with various r bounds policy is prohibited"

        if not self.empirical_r_bounds:
            assert (
                self.rmin == other.rmin
            ), "Summation of estimators with various r bounds is prohibited"
            assert (
                self.rmax == other.rmax
            ), "Summation of estimators with various r bounds is prohibited"

        result = IntervalImpl(
            wmin=min(self.wmin, other.wmin),
            wmax=max(self.wmax, other.wmax),
            rmin=min(self.rmin, other.rmin),
            rmax=max(self.rmax, other.rmax),
            empirical_r_bounds=self.empirical_r_bounds,
        )

        result.n = IncrementalFsum.merge(self.n, other.n)
        result.sumw = IncrementalFsum.merge(self.sumw, other.sumw)
        result.sumwsq = IncrementalFsum.merge(self.sumwsq, other.sumwsq)
        result.sumwr = IncrementalFsum.merge(self.sumwr, other.sumwr)
        result.sumwsqr = IncrementalFsum.merge(self.sumwsqr, other.sumwsqr)
        result.sumwsqrsq = IncrementalFsum.merge(self.sumwsqrsq, other.sumwsqrsq)

        return result


class Estimator(base.Estimator):
    _impl: EstimatorImpl

    def __init__(self, wmin: float = 0, wmax: float = inf):
        self._impl = EstimatorImpl(wmin, wmax)

    def add_example(self, p_log: float, r: float, p_pred: float) -> None:
        self._impl.add(p_pred / p_log, r)

    def get(self) -> Optional[float]:
        return self._impl.get()

    def __add__(self, other: "Estimator") -> "Estimator":
        result = Estimator()
        result._impl = self._impl + other._impl
        return result


class Interval(base.Interval):
    _impl: IntervalImpl

    def __init__(
        self,
        wmin: float = 0,
        wmax: float = inf,
        rmin: float = 0,
        rmax: float = 1,
        empirical_r_bounds: bool = False,
    ) -> None:
        self._impl = IntervalImpl(wmin, wmax, rmin, rmax, empirical_r_bounds)

    def add_example(
        self,
        p_log: float,
        r: float,
        p_pred: float,
        p_drop: float = 0,
        n_drop: Optional[int] = None,
    ) -> None:
        self._impl.add(p_pred / p_log, r, p_drop, n_drop)

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> Tuple[float, float]:
        return self._impl.get(alpha, atol)

    def __add__(self, other: Interval) -> Interval:
        result = Interval()
        result._impl = self._impl + other._impl
        return result
