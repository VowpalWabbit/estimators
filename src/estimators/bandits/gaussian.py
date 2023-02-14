import math
from estimators.bandits import base
from scipy import stats  # type: ignore
from typing import List, Optional, cast


class Interval(base.Interval):
    examples_count: float
    weighted_reward: float
    weighted_reward_sq: float

    def __init__(self) -> None:
        self.examples_count = 0
        self.weighted_reward = 0
        self.weighted_reward_sq = 0

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
        self.examples_count += 1 + n_drop_tmp
        w = p_pred / (p_log * (1 - p_drop))
        self.weighted_reward += r * w
        self.weighted_reward_sq += (r * w) ** 2

    def get(self, alpha: float = 0.05) -> List[Optional[float]]:
        if self.examples_count <= 1:
            return [None, None]

        z_gaussian_cdf = stats.norm.ppf(1 - alpha / 2)
        variance = (
            self.weighted_reward_sq - self.weighted_reward**2 / self.examples_count
        ) / (self.examples_count - 1)
        gauss_delta = z_gaussian_cdf * math.sqrt(max(0, variance) / self.examples_count)
        ips = self.weighted_reward / self.examples_count
        return [ips - gauss_delta, ips + gauss_delta]

    def __add__(self, other: "Interval") -> "Interval":
        result = Interval()
        result.examples_count = self.examples_count + other.examples_count
        result.weighted_reward = self.weighted_reward + other.weighted_reward
        result.weighted_reward_sq = self.weighted_reward_sq + other.weighted_reward_sq
        return result
