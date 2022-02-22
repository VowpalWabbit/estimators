from estimators.bandits import base
from typing import Optional


class Estimator(base.Estimator):
    examples_count: float
    weighted_reward: float

    def __init__(self):
        self.examples_count = 0
        self.weighted_reward = 0

    def add_example(self, p_log: float, r: float, p_pred: float, count: float = 1.0) -> None:
        self.examples_count += count
        w = p_pred / p_log
        self.weighted_reward += r * w * count

    def get(self) -> Optional[float]:
        return self.weighted_reward/self.examples_count if self.examples_count > 0 else None

    def __add__(self, other: 'Estimator') -> 'Estimator':
        result = Estimator()
        result.examples_count = self.examples_count + other.examples_count
        result.weighted_reward = self.weighted_reward + other.weighted_reward
        return result
