from math import inf
from typing import Callable, List, Dict, Optional
import typing
from estimators.bandits.cressieread import EstimatorImpl, IntervalImpl
from estimators.math import IncrementalFsum, clopper_pearson


class Estimator:
    wmin: float
    wmax: float
    n: IncrementalFsum
    _impl: Dict[str, EstimatorImpl]

    def __init__(self, wmin: float = 0, wmax: float = inf):
        self.wmin = wmin
        self.wmax = wmax
        self.n = IncrementalFsum()
        self._impl = {}

    def add_example(
        self,
        slot_ids: List[str],
        p_logs: List[float],
        rs: List[float],
        p_preds: List[float],
    ) -> None:
        if (
            len(p_logs) != len(rs)
            or len(rs) != len(p_preds)
            or len(p_preds) != len(set(slot_ids))
        ):
            raise ValueError(
                f"Error: unique elements in slot_ids and length of p_logs, rs, p_preds must be the same, \
                found {len(set(slot_ids))}, {len(p_logs)}, {len(rs)}, {len(p_preds)}  respectively"
            )

        self.n += 1
        ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
        w = 1.0
        for i in range(len(ws)):
            w *= ws[i]
            if slot_ids[i] not in self._impl:
                self._impl[slot_ids[i]] = EstimatorImpl(0, inf)
            self._impl[slot_ids[i]].add(w, rs[i])

    def get_impression(self) -> Dict[str, float]:
        result = {}
        if float(self.n) > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = float(estimator.n) / float(self.n)
        return result

    def get_r_given_impression(self) -> Dict[str, Optional[float]]:
        result = {}
        if float(self.n) > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = estimator.get()
        return result

    def get_r(self) -> Dict[str, Optional[float]]:
        result = {}
        if float(self.n) > 0:
            impression = self.get_impression()
            r_given_impression = self.get_r_given_impression()
            for slot_id in self._impl.keys():
                value = r_given_impression[slot_id]
                result[slot_id] = (
                    impression[slot_id] * value if value is not None else None
                )
        return result

    def get_r_overall(self) -> Optional[float]:
        return sum(self._impl.values(), EstimatorImpl(0, inf)).get()

    def __add__(self, other: "Estimator") -> "Estimator":
        slot_ids = set(self._impl.keys()).union(set(other._impl.keys()))
        result = Estimator(
            wmin=min(self.wmin, other.wmin), wmax=max(self.wmax, other.wmax)
        )
        result.n = IncrementalFsum.merge(self.n, other.n)
        default: Callable[[], EstimatorImpl] = lambda: EstimatorImpl(0, inf)
        for id in slot_ids:
            result._impl[id] = self._impl.get(id, default()) + other._impl.get(
                id, default()
            )
        return result


class Interval:
    rmin: float
    rmax: float
    n: IncrementalFsum
    _impl: Dict[str, IntervalImpl]

    def __init__(
        self, rmin: float = 0, rmax: float = 1, empirical_r_bounds: bool = False
    ):
        self.rmin = rmin
        self.rmax = rmax
        self.n = IncrementalFsum()
        self._impl = {}
        self.empirical_r_bounds = empirical_r_bounds

    def add_example(
        self,
        slot_ids: List[str],
        p_logs: List[float],
        rs: List[float],
        p_preds: List[float],
        p_drop: float = 0,
        n_drop: Optional[int] = None,
    ) -> None:
        if (
            len(p_logs) != len(rs)
            or len(rs) != len(p_preds)
            or len(p_preds) != len(set(slot_ids))
        ):
            raise ValueError(
                f"Error: unique elements in slot_ids and length of p_logs, rs, p_preds must be the same, \
                found {len(set(slot_ids))}, {len(p_logs)}, {len(rs)}, {len(p_preds)}  respectively"
            )

        self.n += 1
        ws = [p_pred / p_log for p_pred, p_log in zip(p_preds, p_logs)]
        w = 1.0
        for i in range(len(ws)):
            w *= ws[i]
            if slot_ids[i] not in self._impl:
                self._impl[slot_ids[i]] = IntervalImpl(
                    0, inf, self.rmin, self.rmax, self.empirical_r_bounds
                )
            self._impl[slot_ids[i]].add(w, rs[i], p_drop, n_drop)

    # TODO: update return type to Dict[str, Tuple[float]]
    def get_impression(self, alpha: float = 0.05) -> Dict[str, List[float]]:
        result = {}
        # Does self.n > 0 guarantee that each slot has at least one example?
        if float(self.n) > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = clopper_pearson(
                    float(estimator.n), float(self.n), alpha
                )
        return result

    def get_r_given_impression(
        self, alpha: float = 0.05, atol: float = 1e-9
    ) -> Dict[str, List[Optional[float]]]:
        result = {}
        if float(self.n) > 0:
            for slot_id, estimator in self._impl.items():
                result[slot_id] = estimator.get(alpha, atol)
        return result

    def get_r(
        self, alpha: float = 0.05, atol: float = 1e-9
    ) -> Dict[str, List[Optional[float]]]:
        result = {}
        if float(self.n) > 0:
            impression = self.get_impression(alpha)
            r_given_impression = self.get_r_given_impression(alpha, atol)
            for slot_id in self._impl.keys():
                result[slot_id] = [
                    a * b if b is not None else None
                    for a, b in zip(impression[slot_id], r_given_impression[slot_id])
                ]
        return result

    def get_r_overall(
        self, alpha: float = 0.05, atol: float = 1e-9
    ) -> List[Optional[float]]:
        return sum(
            self._impl.values(),
            IntervalImpl(0, inf, self.rmin, self.rmax, self.empirical_r_bounds),
        ).get(alpha, atol)

    def __add__(self, other: "Interval") -> "Interval":
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
        rmin = min(self.rmin, other.rmin)
        rmax = max(self.rmax, other.rmax)
        slot_ids = set(self._impl.keys()).union(set(other._impl.keys()))
        result = Interval(
            rmin=rmin, rmax=rmax, empirical_r_bounds=self.empirical_r_bounds
        )
        result.n = IncrementalFsum.merge(self.n, other.n)
        default: Callable[[], IntervalImpl] = lambda: IntervalImpl(
            wmin=0,
            wmax=inf,
            rmin=rmin,
            rmax=rmax,
            empirical_r_bounds=self.empirical_r_bounds,
        )
        for id in slot_ids:
            result._impl[id] = self._impl.get(id, default()) + other._impl.get(
                id, default()
            )
        return result
