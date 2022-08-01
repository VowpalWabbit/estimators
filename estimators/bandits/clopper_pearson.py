from scipy.stats import beta
from estimators.bandits import base
from typing import List, Optional
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

    def _scale(self, r):
        assert r >= self.rmin and r <= self.rmax, f'Error: {r} is out of [{self.rmin}, {self.rmax}]'
        return (r - self.rmin) / (self.rmax - self.rmin)

    def _scale_back(self, r):
        return self.rmin + r * (self.rmax - self.rmin)

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        assert count == 1.0, "need to explicitly model the pdrop generatively in order to prevent misleading confidence interval widths"
        r = self._scale(r)
        self.examples_count += count
        w = p_pred / p_log
        self.weighted_reward += r * w * count
        self.max_weight = max(self.max_weight, w)

    def get(self, alpha: float = 0.05) -> List[Optional[float]]:
        if self.max_weight > 0.0:
            successes = self.weighted_reward / self.max_weight
            n = self.examples_count / self.max_weight
            cp = clopper_pearson(successes, n, alpha)
            return [self._scale_back(cp[0]), self._scale_back(cp[1])]
        return [None, None]

    def __add__(self, other: 'Interval') -> 'Interval':
        result = Interval()
        result.examples_count = self.examples_count + other.examples_count
        result.weighted_reward = self.weighted_reward + other.weighted_reward
        result.max_weight = max(self.max_weight, other.max_weight)
        return result

