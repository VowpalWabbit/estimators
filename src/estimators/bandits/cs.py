import typing
from estimators.bandits import base
from typing import List, Optional
from estimators.math import IncrementalFsum


def _logwealth(s: float, v: float, rho: float) -> float:
    from math import log

    def loggammalowerinc(*, a: float, x: float) -> float:
        import scipy.special as sc  # type: ignore

        return log(sc.gammainc(a, x)) + typing.cast(float, sc.loggamma(a))

    assert s + v + rho > 0
    assert rho > 0

    return (
        s
        + v
        + rho * log(rho)
        - (v + rho) * log(s + v + rho)
        + loggammalowerinc(a=v + rho, x=s + v + rho)
        - loggammalowerinc(a=rho, x=rho)
    )


def _lblogwealth(t: float, sumXt: float, v: float, rho: float, alpha: float) -> float:
    from math import log
    import scipy.optimize as so  # type: ignore

    assert 0 < alpha < 1, alpha
    thres = -log(alpha)

    minmu = 0
    logwealthminmu = _logwealth(s=sumXt, v=v, rho=rho)

    if logwealthminmu <= thres:
        return minmu

    maxmu = min(1, sumXt / t)
    logwealthmaxmu = _logwealth(s=sumXt - t * maxmu, v=v, rho=rho)

    if logwealthmaxmu >= thres:
        return maxmu

    res = so.root_scalar(
        f=lambda mu: _logwealth(s=sumXt - t * mu, v=v, rho=rho) - thres,
        method="brentq",
        bracket=[minmu, maxmu],
    )
    assert res.converged, res

    # according to docs this is guaranteed to be a float
    return typing.cast(float, res.root)


class IntervalImpl:
    def __init__(self, rmin: float = 0, rmax: float = 1, adjust: bool = True):
        assert rmin <= rmax, (rmin, rmax)

        self.rho = 1
        self.rmin = rmin
        self.rmax = rmax
        self.adjust = adjust

        self.t = 0.0

        self.sumwsqrsq = IncrementalFsum()
        self.sumwsqr = IncrementalFsum()
        self.sumwsq = IncrementalFsum()
        self.sumwr = IncrementalFsum()
        self.sumw = IncrementalFsum()
        self.sumwrxhatlow = IncrementalFsum()
        self.sumwxhatlow = IncrementalFsum()
        self.sumxhatlowsq = IncrementalFsum()
        self.sumwrxhathigh = IncrementalFsum()
        self.sumwxhathigh = IncrementalFsum()
        self.sumxhathighsq = IncrementalFsum()

    def add(self, w: float, r: float, p_drop: float, n_drop: Optional[int]) -> None:
        assert w >= 0
        assert 0 <= p_drop < 1
        assert n_drop is None or n_drop >= 0

        if not self.adjust:
            r = min(self.rmax, max(self.rmin, r))
        else:
            self.rmin = min(self.rmin, r)
            self.rmax = max(self.rmax, r)

        n_drop_tmp: float = (
            float(n_drop) if n_drop is not None else p_drop / (1 - p_drop)
        )
        if n_drop_tmp > 0:
            import scipy.special as sc

            # we have to simulate presenting n_drop_tmp events with w=0 in a row, which we can do in closed form
            # Sum[(a/(b + s))^2, { s, 0, n - 1 }]
            # a^2 PolyGamma[1,b]-a^2 PolyGamma[1,b+n]

            sumXlow = (float(self.sumwr) - float(self.sumw) * self.rmin) / (
                self.rmax - self.rmin
            )
            alow = sumXlow + 1 / 2
            blow = self.t + 1
            self.sumxhatlowsq += alow**2 * (
                sc.polygamma(1, blow).item() - sc.polygamma(1, blow + n_drop_tmp).item()
            )

            sumXhigh = (float(self.sumw) * self.rmax - float(self.sumwr)) / (
                self.rmax - self.rmin
            )
            ahigh = sumXhigh + 1 / 2
            bhigh = self.t + 1
            self.sumxhathighsq += ahigh**2 * (
                sc.polygamma(1, bhigh).item()
                - sc.polygamma(1, bhigh + n_drop_tmp).item()
            )

            # TODO: convert t to float or cast n_drop_tmp to int?
            self.t += n_drop_tmp

        sumXlow = (float(self.sumwr) - float(self.sumw) * self.rmin) / (
            self.rmax - self.rmin
        )
        Xhatlow = (sumXlow + 1 / 2) / (self.t + 1)
        sumXhigh = (float(self.sumw) * self.rmax - float(self.sumwr)) / (
            self.rmax - self.rmin
        )
        Xhathigh = (sumXhigh + 1 / 2) / (self.t + 1)

        w /= 1 - p_drop

        self.sumwsqrsq += (w * r) ** 2
        self.sumwsqr += w**2 * r
        self.sumwsq += w**2
        self.sumwr += w * r
        self.sumw += w
        self.sumwrxhatlow += w * r * Xhatlow
        self.sumwxhatlow += w * Xhatlow
        self.sumxhatlowsq += Xhatlow**2
        self.sumwrxhathigh += w * r * Xhathigh
        self.sumwxhathigh += w * Xhathigh
        self.sumxhathighsq += Xhathigh**2

        self.t += 1

    def get(self, alpha: float) -> List[Optional[float]]:
        if self.t == 0 or self.rmin == self.rmax:
            return [None, None]

        sumvlow = (
            (
                float(self.sumwsqrsq)
                - 2 * self.rmin * float(self.sumwsqr)
                + self.rmin**2 * float(self.sumwsq)
            )
            / (self.rmax - self.rmin) ** 2
            - 2
            * (float(self.sumwrxhatlow) - self.rmin * float(self.sumwxhatlow))
            / (self.rmax - self.rmin)
            + float(self.sumxhatlowsq)
        )
        sumXlow = (float(self.sumwr) - float(self.sumw) * self.rmin) / (
            self.rmax - self.rmin
        )
        l = _lblogwealth(
            t=self.t, sumXt=sumXlow, v=sumvlow, rho=self.rho, alpha=alpha / 2
        )

        sumvhigh = (
            (
                float(self.sumwsqrsq)
                - 2 * self.rmax * float(self.sumwsqr)
                + self.rmax**2 * float(self.sumwsq)
            )
            / (self.rmax - self.rmin) ** 2
            + 2
            * (float(self.sumwrxhathigh) - self.rmax * float(self.sumwxhathigh))
            / (self.rmax - self.rmin)
            + float(self.sumxhathighsq)
        )
        sumXhigh = (float(self.sumw) * self.rmax - float(self.sumwr)) / (
            self.rmax - self.rmin
        )
        u = 1 - _lblogwealth(
            t=self.t, sumXt=sumXhigh, v=sumvhigh, rho=self.rho, alpha=alpha / 2
        )

        return [
            self.rmin + l * (self.rmax - self.rmin),
            self.rmin + u * (self.rmax - self.rmin),
        ]


class Interval(base.Interval):
    _impl: IntervalImpl

    def __init__(
        self, rmin: float = 0, rmax: float = 1, empirical_r_bounds: bool = False
    ) -> None:
        self._impl = IntervalImpl(rmin=rmin, rmax=rmax, adjust=empirical_r_bounds)

    def add_example(
        self,
        p_log: float,
        r: float,
        p_pred: float,
        p_drop: float = 0,
        n_drop: Optional[int] = None,
    ) -> None:
        self._impl.add(p_pred / p_log, r, p_drop, n_drop)

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> List[Optional[float]]:
        return self._impl.get(alpha)

    def __add__(self, other: "Interval") -> "Interval":
        raise NotImplementedError
