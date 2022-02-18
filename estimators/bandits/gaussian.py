import math
from estimators.bandits import base
from scipy import stats
from typing import List


class Interval(base.Interval):
    examples_count: float
    weighted_reward: float
    weighted_reward_sq: float

    def __init__(self):
        self.examples_count = 0
        self.weighted_reward = 0
        self.weighted_reward_sq = 0

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        self.examples_count += count
        w = p_pred/p_log
        self.weighted_reward += r * w * count
        self.weighted_reward_sq += ((r * w)**2) * count

    def get(self, alpha: float = 0.05) -> List[float]:
        if self.weighted_reward_sq > 0.0 and self.examples_count > 1:
            z_gaussian_cdf = stats.norm.ppf(1 - alpha / 2)
            variance = (self.weighted_reward_sq - self.weighted_reward**2 / self.examples_count) / \
                       (self.examples_count - 1)
            gauss_delta = z_gaussian_cdf * math.sqrt(variance / self.examples_count)
            ips = self.weighted_reward / self.examples_count
            return [ips - gauss_delta, ips + gauss_delta]
        return [None, None]
