from estimators.ccb import base
from math import inf
from typing import List, Callable
from estimators.bandits.cressieread import EstimatorImpl, IntervalImpl


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
    _impl_ctr: Callable[[int], IntervalImpl]
    _impl: List[IntervalImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1):
        self._impl_ctr = lambda step: IntervalImpl(wmin, wmax, rmin, rmax, step)
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


    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> List[List[float]]:
        result = []
        n0 = float(self._impl[0].n)
        if n0 > 0:
            for impl in self._impl:
                result.append([v * float(impl.n) / n0 for v in impl.get()])
        return result
