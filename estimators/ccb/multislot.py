from math import inf
from scipy.stats import beta
from typing import List, Callable, Dict
from estimators.bandits.cressieread import EstimatorImpl, IntervalImpl
from estimators.math import IncrementalFsum


class Estimator():
    wmin: float
    wmax: float
    n: IncrementalFsum
    _impl: Dict[str, EstimatorImpl]


    def __init__(self, wmin: float = 0, wmax: float = inf):
        self.wmin = wmin
        self.wmax = wmax
        self.n = IncrementalFsum()
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

    def get_appearance(self) -> Dict[str, float]:
        result = {}
        if float(self.n) > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = float(estimator.n) / float(self.n)
        return result

    def get_conditional(self) -> Dict[str, float]:
        result = {}
        if float(self.n) > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = estimator.get()
        return result

    def get(self) -> Dict[str, float]:
        result = {}
        if float(self.n) > 0:
            appearance = self.get_appearance()
            conditional = self.get_conditional()
            for slot_id, estimator in self._impl.items():
                result[slot_id] = appearance[slot_id] * conditional[slot_id]
        return result


class Interval():
    wmin: float
    wmax: float
    rmin: float
    rmax: float
    n: IncrementalFsum
    _impl: Dict[str, IntervalImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf, rmin: float = 0, rmax: float = 1, empirical_r_bounds: bool = False):
        self.wmin = wmin
        self.wmax = wmax
        self.rmin = rmin
        self.rmax = rmax
        self.n = IncrementalFsum()
        self._impl = {}
        self.empirical_r_bounds = empirical_r_bounds

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
                    self._impl[slot_ids[i]] = IntervalImpl(0, inf, self.rmin, self.rmax, self.empirical_r_bounds)
                self._impl[slot_ids[i]].add(w, rs[i], count)

    def get_appearance(self, alpha: float = 0.05) -> Dict[str, List[float]]:
        result = {}
        if float(self.n) > 0:
            for slot_id, estimator in self._impl.items():
                lower_bound = beta.ppf(alpha / 2, float(estimator.n), float(self.n) - float(estimator.n) + 1)
                upper_bound = beta.ppf(1 - alpha / 2, float(estimator.n) + 1, float(self.n) - float(estimator.n))
                if int(float(estimator.n)) == 0:
                    lower_bound = 0
                if int(float(estimator.n)) == int(float(self.n)):
                    upper_bound = 1
                result[slot_id] = [lower_bound, upper_bound]
        return result

    def get_conditional(self, alpha: float = 0.05, atol: float = 1e-9) -> Dict[str, List[float]]:
        result = {}
        if float(self.n) > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = estimator.get(alpha, atol)
        return result

    def get(self, alpha: float = 0.05, atol: float = 1e-9) -> Dict[str, List[float]]:
        result = {}
        if float(self.n) > 0:
            appearance = self.get_appearance(alpha)
            conditional = self.get_conditional(alpha, atol)
            for slot_id, estimator in self._impl.items():
                result[slot_id] = [a * b for a, b in zip(appearance[slot_id], conditional[slot_id])]
        return result
