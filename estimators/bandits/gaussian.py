import math
from estimators.bandits import base
from scipy import stats
from typing import List, Optional


class Interval(base.Interval):
    examples_count: float
    weighted_reward: float
    weighted_reward_sq: float

    def __init__(self):
        self.examples_count = 0
        self.weighted_reward = 0
        self.weighted_reward_sq = 0

    def add_example(self, p_log: float, r: float, p_pred: float, count: int = 1) -> None:
        self.examples_count += count
        w = p_pred/p_log
        self.weighted_reward += r * w * count
        self.weighted_reward_sq += ((r * w)**2) * count

    def get(self, alpha: float = 0.05) -> List[Optional[float]]:
        if self.examples_count <= 1:
            return [None, None]

        z_gaussian_cdf = stats.norm.ppf(1 - alpha / 2)
        variance = (self.weighted_reward_sq - self.weighted_reward**2 / self.examples_count) / \
                   (self.examples_count - 1)
        gauss_delta = z_gaussian_cdf * math.sqrt(variance / self.examples_count)
        ips = self.weighted_reward / self.examples_count
        return [ips - gauss_delta, ips + gauss_delta]

    def __add__(self, other: 'Interval') -> 'Interval':
        result = Interval()
        result.examples_count = self.examples_count + other.examples_count
        result.weighted_reward = self.weighted_reward + other.weighted_reward
        result.weighted_reward_sq = self.weighted_reward_sq + other.weighted_reward_sq
        return result
