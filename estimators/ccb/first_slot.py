from estimators.ccb import base
from typing import List


class Estimator(base.Estimator):
    def __init__(self, bandits_impl):
        self.impl = bandits_impl
        self.slots_count = 0

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        """Expects lists for logged probabilities, rewards and predicted probabilities. These should correspond to
        each slot. """

        if not isinstance(p_logs, list) and not isinstance(rs, list) and not isinstance(p_preds, list):
            raise ValueError('Error: p_logs, r and p_preds must be lists')

        if len(p_logs) != len(p_preds) and len(p_logs) != len(rs) and len(rs) != len(p_preds):
            raise ValueError(f'Error: p_logs, r and p_preds must be the same length, found {len(p_logs)}, {len(rs)} and'
                             f'{len(p_preds)} respectively')
        self.slots_count = max(self.slots_count, len(p_logs))
        self.impl.add_example(p_logs[0], rs[0], p_preds[0])

    def get(self) -> List[float]:
        if self.slots_count > 0:
            result = self.impl.get()
            return [result] + [0] * (self.slots_count - 1)
        return []


class Interval(base.Interval):
    def __init__(self, bandits_impl):
        self.impl = bandits_impl
        self.slots_count = 0

    def add_example(self, p_logs: List[float], rs: List[float], p_preds: List[float], count: float = 1.0) -> None:
        """Expects lists for logged probabilities, rewards and predicted probabilities. These should correspond to
        each slot. """

        if not isinstance(p_logs, list) and not isinstance(rs, list) and not isinstance(p_preds, list):
            raise ValueError('Error: p_logs, r and p_preds must be lists')

        if len(p_logs) != len(p_preds) and len(p_logs) != len(rs) and len(rs) != len(p_preds):
            raise ValueError(f'Error: p_logs, r and p_preds must be the same length, found {len(p_logs)}, {len(rs)} and'
                             f'{len(p_preds)} respectively')
        self.slots_count = max(self.slots_count, len(p_logs))
        self.impl.add_example(p_logs[0], rs[0], p_preds[0])

    def get(self, alpha: float = 0.05) -> List[List[float]]:
        if self.slots_count > 0:
            result = self.impl.get()
            return [result] + [[0, 0]] * (self.slots_count - 1)
        return []
