from __future__ import annotations

from estimators.bandits import base
from typing import Optional


class Estimator(base.Estimator):
    weighted_examples_count: float
    weighted_reward: float

    def __init__(self) -> None:
        self.weighted_examples_count = 0
        self.weighted_reward = 0

    def add_example(self, p_log: float, r: float, p_pred: float) -> None:
        w = p_pred / p_log
        self.weighted_examples_count += w
        self.weighted_reward += r * w

    def get(self) -> Optional[float]:
        return (
            self.weighted_reward / self.weighted_examples_count
            if self.weighted_examples_count != 0
            else None
        )

    def __add__(self, other: Estimator) -> Estimator:
        result = Estimator()
        result.weighted_examples_count = (
            self.weighted_examples_count + other.weighted_examples_count
        )
        result.weighted_reward = self.weighted_reward + other.weighted_reward
        return result
