from estimators.ccb import base
from math import inf
from typing import List, Callable, Dict
from estimators.bandits.cressieread import EstimatorImpl, IntervalImpl


class Estimator(base.Estimator):
    wmin: float
    wmax: float
    _impl: Dict[str, EstimatorImpl]


    def __init__(self, wmin: float = 0, wmax: float = inf):
        self.wmin = wmin
        self.wmax = wmax
        self._impl = {}

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0, slot_ids: List[str] = None) -> None:
        if len(p_logs) != len(rs) or len(rs) != len(p_preds):
            raise ValueError(f'Error: p_logs, r and p_preds must be the same length, found {len(p_logs)}, {len(rs)} and {len(p_preds)} respectively')

        if slot_ids != None and set(slot_ids) != len(p_logs):
            raise ValueError(f'Error: unique elements in slots_id and len of p_logs must be same, found {len(slot_ids)}, {len(p_logs)} respectively')

        if count > 0:
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            w = 1.0
            slot_ids = slot_ids if slot_ids != None else [str(x) for x in range(len(ws))]
            for i in range(len(ws)):
                w *= ws[i]
                if slot_ids[i] not in self._impl:
                    self._impl[slot_ids[i]] = EstimatorImpl(0, inf)
                    # self._impl[slot_ids[i]] = EstimatorImpl(self.wmin ** (i+1), self.wmax ** (i+1)) #for reference
                self._impl[slot_ids[i]].add(w, rs[i], count)

    def get(self) -> Dict[str, float]:
        result = {}
        n0 = max([float(impl.n) for impl in self._impl.values()], default=0)
        if n0 > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = estimator.get() * float(estimator.n) / n0
        return result


class Interval(base.Interval):
    wmin: float
    wmax: float
    rmin: float
    rmax: float
    _impl: Dict[str, IntervalImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1):
        self.wmin = wmin
        self.wmax = wmax
        self.rmin = rmin
        self.rmax = rmax
        self._impl = {}

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0, slot_ids: List[str] = None) -> None:
        if len(p_logs) != len(rs) or len(rs) != len(p_preds):
            raise ValueError(f'Error: p_logs, r and p_preds must be the same length, found {len(p_logs)}, {len(rs)} and {len(p_preds)} respectively')

        if slot_ids != None and set(slot_ids) != len(p_logs):
            raise ValueError(f'Error: slots_id if specified and p_logs must be the same length, found {len(slot_ids)}, {len(p_logs)} respectively')

        if count > 0:
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            w = 1.0
            slot_ids = slots if slot_ids != None else [str(x) for x in range(len(ws))]
            for i in range(len(ws)):
                w *= ws[i]
                if slot_ids[i] not in self._impl:
                    self._impl[slot_ids[i]] = IntervalImpl(0, inf, self.rmin, self.rmax)
                    #self._impl[slot_ids[i]] = IntervalImpl(self.wmin ** (i+1), self.wmax ** (i+1), self.rmin, self.rmax) #for reference
                self._impl[slot_ids[i]].add(w, rs[i], count)

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> Dict[str, List[float]]:
        result = {}
        n0 = max([float(impl.n) for impl in self._impl.values()], default=0)
        if n0 > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = [v * float(estimator.n) / n0 for v in estimator.get()]
        return result
