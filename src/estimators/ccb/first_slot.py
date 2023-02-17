from __future__ import annotations

from estimators.ccb import base
from estimators.bandits import base as bandits_base
from typing import List, Optional, Tuple
from copy import deepcopy


class Estimator(base.Estimator):
    def __init__(self, bandits_impl: bandits_base.Estimator) -> None:
        self.impl = bandits_impl
        self.slots_count = 0

    def add_example(
        self, p_logs: List[float], rs: List[float], p_preds: List[float]
    ) -> None:
        """Expects lists for logged probabilities, rewards and predicted probabilities. These should correspond to
        each slot."""

        if (
            not isinstance(p_logs, list)
            and not isinstance(rs, list)
            and not isinstance(p_preds, list)
        ):
            raise ValueError("Error: p_logs, r and p_preds must be lists")

        if (
            len(p_logs) != len(p_preds)
            and len(p_logs) != len(rs)
            and len(rs) != len(p_preds)
        ):
            raise ValueError(
                f"Error: p_logs, r and p_preds must be the same length, found {len(p_logs)}, {len(rs)} and"
                f"{len(p_preds)} respectively"
            )
        self.slots_count = max(self.slots_count, len(p_logs))
        self.impl.add_example(p_logs[0], rs[0], p_preds[0])

    def get_impression(self) -> List[float]:
        if self.slots_count > 0:
            return [1.0] + [0] * (self.slots_count - 1)
        return []

    def get_r_given_impression(self) -> List[Optional[float]]:
        if self.slots_count > 0:
            result = self.impl.get()
            return [result] + [0] * (self.slots_count - 1)
        return []

    def get_r(self) -> List[Optional[float]]:
        return self.get_r_given_impression()

    def get_r_overall(self) -> Optional[float]:
        return self.get_r()[0]

    def __add__(self, other: Estimator) -> Estimator:
        # TODO: dependency on instance -> dependency on factory
        raise NotImplementedError()


class Interval(base.Interval):
    def __init__(self, bandits_impl: bandits_base.Interval) -> None:
        self.impl = bandits_impl
        self.slots_count = 0

    def add_example(
        self,
        p_logs: List[float],
        rs: List[float],
        p_preds: List[float],
        p_drop: float = 0,
        n_drop: Optional[int] = None,
    ) -> None:
        """Expects lists for logged probabilities, rewards and predicted probabilities. These should correspond to
        each slot."""

        if (
            not isinstance(p_logs, list)
            and not isinstance(rs, list)
            and not isinstance(p_preds, list)
        ):
            raise ValueError("Error: p_logs, r and p_preds must be lists")

        if (
            len(p_logs) != len(p_preds)
            and len(p_logs) != len(rs)
            and len(rs) != len(p_preds)
        ):
            raise ValueError(
                f"Error: p_logs, r and p_preds must be the same length, found {len(p_logs)}, {len(rs)} and"
                f"{len(p_preds)} respectively"
            )
        self.slots_count = max(self.slots_count, len(p_logs))
        self.impl.add_example(p_logs[0], rs[0], p_preds[0], p_drop, n_drop)

    def get_impression(self, alpha: float = 0.05) -> List[Tuple[float, float]]:
        if self.slots_count > 0:
            result = self.impl.get(alpha)
            return [(1.0, 1.0)] + [(0.0, 0.0)] * (self.slots_count - 1)
        return []

    def get_r_given_impression(self, alpha: float = 0.05) -> List[Tuple[float, float]]:
        if self.slots_count > 0:
            result = self.impl.get(alpha)
            return [result] + [(0.0, 0.0)] * (self.slots_count - 1)
        return []

    def get_r(self, alpha: float = 0.05) -> List[Tuple[float, float]]:
        return self.get_r_given_impression(alpha)

    def get_r_overall(self, alpha: float = 0.05) -> Tuple[float, float]:
        return self.get_r(alpha)[0]

    def __add__(self, other: Interval) -> Interval:
        raise NotImplementedError()
