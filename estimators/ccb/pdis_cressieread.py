from estimators.ccb import base
from math import inf
from typing import List, Callable
from estimators.bandits.cressieread import EstimatorImpl, IntervalImpl
from copy import deepcopy


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

    def __add__(self, other: 'Estimator') -> 'Estimator':
        (large, small) = (self, other) if len(self._impl) >= len(other._impl) else (other, self)
        result = Estimator(wmin = min(self.wmin, other.wmin), wmax = max(self.wmax, other.wmax))
        for i in range(len(large._impl)):
            result._impl.append(large._impl[i] + small._impl[i] if i < len(small._impl) else deepcopy(large._impl[i]))
        return result


class Interval(base.Interval):
    wmin: float
    wmax: float
    rmin: float
    rmax: float
    _impl: List[IntervalImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1, empirical_r_bounds: bool = False):
        self.wmin = wmin
        self.wmax = wmax
        self.rmin = rmin
        self.rmax = rmax
        self._impl = []
        self.empirical_r_bounds = empirical_r_bounds

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        if count > 0:
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            w = 1.0
            for i in range(len(ws)):
                w *= ws[i]
                if len(self._impl) <= i:
                    self._impl.append(IntervalImpl(self.wmin ** (i + 1), self.wmax ** (i + 1), self.rmin, self.rmax, self.empirical_r_bounds))
                self._impl[i].add(w, rs[i], count)

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> List[List[float]]:
        result = []
        n0 = float(self._impl[0].n) if any(self._impl) else 0
        if n0 > 0:
            for impl in self._impl:
                result.append([v * float(impl.n) / n0 for v in impl.get()])
        return result

    def __add__(self, other: 'Interval') -> 'Interval':
        assert not (self.empirical_r_bounds ^ other.empirical_r_bounds), 'Summation of estimators with various r bounds policy is prohibited'
        
        if not self.empirical_r_bounds:
            assert self.rmin == other.rmin, 'Summation of estimators with various r bounds is prohibited'
            assert self.rmax == other.rmax, 'Summation of estimators with various r bounds is prohibited'

        (large, small) = (self, other) if len(self._impl) >= len(other._impl) else (other, self)
        result = Interval(
            wmin = min(self.wmin, other.wmin),
            wmax=max(self.wmax, other.wmax),
            rmin=min(self.rmin, other.rmin),
            rmax=max(self.rmax, other.rmax),
            empirical_r_bounds=self.empirical_r_bounds)
        for i in range(len(large._impl)):
            result._impl.append(large._impl[i] + small._impl[i] if i < len(small._impl) else deepcopy(large._impl[i]))
        return result
