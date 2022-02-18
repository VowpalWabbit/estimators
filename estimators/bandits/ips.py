from estimators.bandits import base


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

    def get(self) -> float:
        if self.examples_count == 0:
            raise ValueError('Error: No data point added')

        return self.weighted_reward/self.examples_count
