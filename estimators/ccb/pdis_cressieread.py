from estimators.ccb import base
from math import inf
from typing import List, Optional
from estimators.bandits.cressieread import EstimatorImpl, IntervalImpl
from estimators.math import clopper_pearson
from copy import deepcopy


class Estimator(base.Estimator):
    wmin: float
    wmax: float
    _impl: List[EstimatorImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf):
        self.wmin = wmin
        self.wmax = wmax
        self._impl = []

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float]) -> None:
        ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
        w = 1.0
        for i in range(len(ws)):
            w *= ws[i]
            if len(self._impl) <= i:
                self._impl.append(EstimatorImpl(self.wmin ** (i + 1), self.wmax ** (i + 1)))
            self._impl[i].add(w, rs[i])

    def get_impression(self) -> List[float]:
        total = float(self._impl[0].n) if any(self._impl) else 0
        return [float(e.n) / total for e in self._impl]

    def get_r_given_impression(self) -> List[float]:
        return [e.get() for e in self._impl]

    def get_r(self) -> List[float]:
        return [e[0] * e[1] for e in zip(self.get_impression(), self.get_r_given_impression())]

    def get_r_overall(self) -> float:
        return sum(self._impl, EstimatorImpl(0, inf)).get()        

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

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], p_drop: float = 0, n_drop: Optional[int] = None) -> None:
        ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
        w = 1.0
        for i in range(len(ws)):
            w *= ws[i]
            if len(self._impl) <= i:
                self._impl.append(IntervalImpl(self.wmin ** (i + 1), self.wmax ** (i + 1), self.rmin, self.rmax, self.empirical_r_bounds))
            self._impl[i].add(w, rs[i], p_drop, n_drop)

    def get_impression(self, alpha: float = 0.05) -> List[List[float]]:
        total = float(self._impl[0].n) if any(self._impl) else 0
        return [clopper_pearson(float(e.n), total, alpha) for e in self._impl]

    def get_r_given_impression(self, alpha: float = 0.05) -> List[List[float]]:
        return [e.get(alpha) for e in self._impl]

    def get_r(self, alpha: float = 0.05) -> List[List[float]]:
        return [[e[0][0] * e[1][0], e[0][1] * e[1][1]] for e in zip(self.get_impression(alpha), self.get_r_given_impression(alpha))]

    def get_r_overall(self, alpha: float = 0.05) -> List[float]:
        return sum(self._impl, IntervalImpl(0, inf, self.rmin, self.rmax)).get(alpha)   

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
