from math import inf
from typing import List, Callable, Dict
from estimators.bandits.cressieread import EstimatorImpl, IntervalImpl
from estimators.math import IncrementalFsum


class Estimator():
    wmin: float
    wmax: float
    n: float
    _impl: Dict[str, EstimatorImpl]


    def __init__(self, wmin: float = 0, wmax: float = inf):
        self.wmin = wmin
        self.wmax = wmax
        self.n = 0
        self._impl = {}

    def add_example(self, slot_ids: List[str], p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        if len(p_logs) != len(rs) or len(rs) != len(p_preds) or len(p_preds) != len(set(slot_ids)):
            raise ValueError(f'Error: unique elements in slot_ids and length of p_logs, rs, p_preds must be the same, \
                found {len(set(slot_ids))}, {len(p_logs)}, {len(rs)}, {len(p_preds)}  respectively')

        if count > 0:
            self.n += count
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            w = 1.0
            for i in range(len(ws)):
                w *= ws[i]
                if slot_ids[i] not in self._impl:
                    self._impl[slot_ids[i]] = EstimatorImpl(0, inf)
                self._impl[slot_ids[i]].add(w, rs[i], count)

    def get(self) -> Dict[str, float]:
        result = {}
        if self.n > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = estimator.get() * float(estimator.n) / self.n
        return result


class Interval():
    wmin: float
    wmax: float
    rmin: float
    rmax: float
    n: float
    _impl: Dict[str, IntervalImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1):
        self.wmin = wmin
        self.wmax = wmax
        self.rmin = rmin
        self.rmax = rmax
        self.n = 0
        self._impl = {}

    def add_example(self, slot_ids: List[str], p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        if len(p_logs) != len(rs) or len(rs) != len(p_preds) or len(p_preds) != len(set(slot_ids)):
            raise ValueError(f'Error: unique elements in slot_ids and length of p_logs, rs, p_preds must be the same, \
                found {len(set(slot_ids))}, {len(p_logs)}, {len(rs)}, {len(p_preds)}  respectively')

        if count > 0:
            self.n += count
            ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
            w = 1.0
            for i in range(len(ws)):
                w *= ws[i]
                if slot_ids[i] not in self._impl:
                    self._impl[slot_ids[i]] = IntervalImpl(0, inf, self.rmin, self.rmax)
                self._impl[slot_ids[i]].add(w, rs[i], count)

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> Dict[str, List[float]]:
        result = {}
        if self.n > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = [v * float(estimator.n) / self.n for v in estimator.get()]
        return result
