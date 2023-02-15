from estimators.bandits import base
from typing import List, Optional, Tuple
from estimators.math import clopper_pearson


class Interval(base.Interval):
    examples_count: float
    weighted_reward: float
    max_weight: float

    def __init__(self, rmin: float = 0, rmax: float = 1):
        self.examples_count = 0
        self.weighted_reward = 0
        self.max_weight = 0
        self.rmin = rmin
        self.rmax = rmax

    def _scale(self, r: float) -> float:
        assert (
            r >= self.rmin and r <= self.rmax
        ), f"Error: {r} is out of [{self.rmin}, {self.rmax}]"
        return (r - self.rmin) / (self.rmax - self.rmin)

    def _scale_back(self, r: float) -> float:
        return self.rmin + r * (self.rmax - self.rmin)

    def add_example(
        self,
        p_log: float,
        r: float,
        p_pred: float,
        p_drop: float = 0,
        n_drop: Optional[int] = None,
    ) -> None:
        n_drop_tmp: float = (
            float(n_drop) if n_drop is not None else p_drop / (1 - p_drop)
        )
        r = self._scale(r)
        self.examples_count += 1 + n_drop_tmp
        w = p_pred / (p_log * (1 - p_drop))
        self.weighted_reward += r * w
        self.max_weight = max(self.max_weight, w)

    def get(self, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
        if self.max_weight > 0.0:
            successes = self.weighted_reward / self.max_weight
            n = self.examples_count / self.max_weight
            cp = clopper_pearson(successes, n, alpha)
            return (self._scale_back(cp[0]), self._scale_back(cp[1]))
        return (None, None)

    def __add__(self, other: "Interval") -> "Interval":
        result = Interval()
        result.examples_count = self.examples_count + other.examples_count
        result.weighted_reward = self.weighted_reward + other.weighted_reward
        result.max_weight = max(self.max_weight, other.max_weight)
        return result
