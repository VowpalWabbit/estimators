import math
from estimators.slates import base
from scipy import stats  # type: ignore
from typing import List, Optional


class Interval(base.Interval):
    examples_count: float
    weighted_reward: float
    weighted_reward_sq: float

    def __init__(self) -> None:
        self.examples_count = 0
        self.weighted_reward = 0
        self.weighted_reward_sq = 0

    def add_example(
        self, p_logs: List[float], r: float, p_preds: List[float], count: float = 1.0
    ) -> None:
        """Expects lists for logged probabilities and predicted probabilities. These should correspond to each slot.
        This function is implemented under the simplifying assumptions of
        example 4 in the paper 'Off-policy evaluation for slate recommendation'
        where the slate space is a cartesian product and the logging policy is a
        product distribution"""
        if not isinstance(p_logs, list) or not isinstance(p_preds, list):
            raise ValueError("Error: p_logs and p_preds must be lists")

        if len(p_logs) != len(p_preds):
            raise ValueError(
                f"Error: p_logs and p_preds must be the same length, found {len(p_logs)} "
                f"and {len(p_preds)} respectively"
            )

        self.examples_count += count
        num_slots = len(p_logs)
        w = 1.0 - num_slots
        for p_log, p_pred in zip(p_logs, p_preds):
            w += p_pred / p_log
        self.weighted_reward += r * w * count
        self.weighted_reward_sq += ((r * w) ** 2) * count

    def get(self, alpha: float = 0.05) -> List[Optional[float]]:
        if self.examples_count <= 1:
            return [None, None]

        z_gaussian_cdf = stats.norm.ppf(1 - alpha / 2)
        variance = (
            self.weighted_reward_sq
            - self.weighted_reward * self.weighted_reward / self.examples_count
        ) / (self.examples_count - 1)
        gauss_delta = z_gaussian_cdf * math.sqrt(variance / self.examples_count)
        ips = self.weighted_reward / self.examples_count
        return [ips - gauss_delta, ips + gauss_delta]

    def __add__(self, other: "Interval") -> "Interval":
        result = Interval()
        result.examples_count = self.examples_count + other.examples_count
        result.weighted_reward = self.weighted_reward + other.weighted_reward
        result.weighted_reward_sq = self.weighted_reward_sq + other.weighted_reward_sq
        return result
