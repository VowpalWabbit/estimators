from sre_constants import AT_LOCALE
from scipy.stats import beta
from estimators.bandits import base
from typing import List, Optional


class Interval(base.Interval):
    examples_count: float
    weighted_reward: float
    max_weighted_reward: float

    def __init__(self, rmin: float = 0, rmax: float = 1):
        self.examples_count = 0
        self.weighted_reward = 0
        self.max_weighted_reward = 0
        self.rmin = rmin
        self.rmax = rmax

    def _scale(self, r):
        assert r >= self.rmin and r <= self.rmax, f'Error: {r} is out of [{self.rmin}, {self.rmax}]'
        return (r - self.rmin) / (self.rmax - self.rmin)

    def _scale_back(self, r):
        return self.rmin + r * (self.rmax - self.rmin)

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        r = self._scale(r)
        self.examples_count += count
        w = p_pred / p_log
        self.weighted_reward += r * w * count
        self.max_weighted_reward = max(self.max_weighted_reward, r * w)

    def get(self, alpha: float = 0.05) -> List[Optional[float]]:
        atol = 1e-10
        if self.max_weighted_reward > 0.0:
            successes = self.weighted_reward / self.max_weighted_reward
            n = self.examples_count / self.max_weighted_reward
            return [self._scale_back(beta.ppf(alpha / 2, successes, max(n - successes + 1, atol))),
                    self._scale_back(beta.ppf(1 - alpha / 2, successes + 1, max(n - successes, atol)))]
        return [None, None]

    def __add__(self, other: 'Interval') -> 'Interval':
        result = Interval()
        result.examples_count = self.examples_count + other.examples_count
        result.weighted_reward = self.weighted_reward + other.weighted_reward
        result.max_weighted_reward = max(self.max_weighted_reward, other.max_weighted_reward)
        return result

