from estimators.slates import base
from typing import List, Optional

# PseudoInverse estimator for slate recommendation. The following implements the
# case for a Cartesian product when mu is a product distribution. This can be
# seen in example 4 of the paper.
# https://arxiv.org/abs/1605.04812


class Estimator(base.Estimator):
    examples_count: float
    weighted_reward: float

    def __init__(self):
        self.examples_count = 0
        self.weighted_reward = 0

    def add_example(self, p_logs: List[float], r: float, p_preds: List[float], count: float = 1.0) -> None:
        """Expects lists for logged probabilities and predicted probabilities. These should correspond to each slot.
        This function is implemented under the simplifying assumptions of
        example 4 in the paper 'Off-policy evaluation for slate recommendation'
        where the slate space is a cartesian product and the logging policy is a
        product distribution"""
        if not isinstance(p_logs, list) or not isinstance(p_preds, list):
            raise ValueError('Error: p_logs and p_preds must be lists')

        if(len(p_logs) != len(p_preds)):
            raise ValueError('Error: p_logs and p_preds must be the same length, found {} and {} respectively'.format(len(p_logs), len(p_preds)))

        self.examples_count += count
        num_slots = len(p_logs)
        w = 1 - num_slots
        for p_log, p_pred in zip(p_logs, p_preds):
            w += p_pred/p_log
        self.weighted_reward += r * w * count

    def get(self) -> Optional[float]:
        if self.examples_count > 0:
            return self.weighted_reward / self.examples_count
        return None
