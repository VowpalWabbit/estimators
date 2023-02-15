""" Interface for implementation of conditional contextual bandits estimators """

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class Estimator(ABC):
    """Interface for implementation of conditional contextual bandits estimators"""

    @abstractmethod
    def add_example(
        self, p_log: List[float], r: List[float], p_pred: List[float]
    ) -> None:
        """
        Args:
                p_log: List of probability of the logging policy
                r: List of reward for choosing an action in the given context
                p_pred: List of predicted probability of making decision
        """
        ...

    @abstractmethod
    def get(self) -> List[Optional[float]]:
        """Calculates the selected estimator. The length of the list is the maximum number of slots seen so far. Estimations per slot are calculated by CB estimator.

        Returns:
                The estimator value.
        """
        ...


class Interval(ABC):
    """Interface for implementation of conditional contextual bandits estimators interval"""

    @abstractmethod
    def add_example(
        self,
        p_log: List[float],
        r: List[float],
        p_pred: List[float],
        p_drop: float = 0,
        n_drop: Optional[int] = None,
    ) -> None:
        """
        Args:
                p_log: List of probability of the logging policy
                r: List of reward for choosing an action in the given context
                p_pred: List of predicted probability of making decision
                p_drop: probability for event to be dropped
                n_drop: amount of dropped events between current and previous ones (populated as p_drop/(1-p_drop) if None)
        """
        ...

    @abstractmethod
    def get(self, alpha: float) -> List[Tuple[Optional[float], Optional[float]]]:
        """Calculates the CI
        Args:
                alpha: alpha value
        Returns:
                Returns the confidence interval as list[float]
        """
        ...
