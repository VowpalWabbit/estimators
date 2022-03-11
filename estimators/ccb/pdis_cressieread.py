from estimators.ccb import base
from math import inf
from typing import List, Callable
from estimators.bandits.cressieread import EstimatorImpl, IntervalImpl


class Estimator(base.Estimator):
    wmin: float
    wmax: float
    _impl: List[EstimatorImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf):
        self.wmin = wmin
        self.wmax = wmax
        self._impl = []

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        if count > 0:
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            w = 1.0
            for i in range(len(ws)):
                w *= ws[i]
                if len(self._impl) <= i:
                    self._impl.append(EstimatorImpl(self.wmin ** (i + 1), self.wmax ** (i + 1)))
                self._impl[i].add(w, rs[i], count)

    def get(self) -> List[float]:
        result = []
        n0 = float(self._impl[0].n) if any(self._impl) else 0
        if n0 > 0:
            for impl in self._impl:
                result.append(impl.get() * float(impl.n) / n0)
        return result


class Interval(base.Interval):
    wmin: float
    wmax: float
    rmin: float
    rmax: float
    _impl: List[IntervalImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1):
        self.wmin = wmin
        self.wmax = wmax
        self.rmin = rmin
        self.rmax = rmax
        self._impl = []

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        if count > 0:
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            w = 1.0
            for i in range(len(ws)):
                w *= ws[i]
                if len(self._impl) <= i:
                    self._impl.append(IntervalImpl(self.wmin ** (i + 1), self.wmax ** (i + 1), self.rmin, self.rmax, True))
                self._impl[i].add(w, rs[i], count)

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> List[List[float]]:
        result = []
        n0 = float(self._impl[0].n) if any(self._impl) else 0
        if n0 > 0:
            for impl in self._impl:
                result.append([v * float(impl.n) / n0 for v in impl.get()])
        return result
