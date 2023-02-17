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
    def get_impression(self) -> List[float]:
        """Calculates impression probability per slot. The length of the list is the maximum number of slots seen so far.

        Returns:
                Array of estimated probabilities
        """
        ...

    @abstractmethod
    def get_r_given_impression(self) -> List[Optional[float]]:
        """Calculates estimated reward per slot conditioned on impression. The length of the list is the maximum number of slots seen so far. Estimations per slot are calculated by CB estimator.

        Returns:
                Array of estimated reward values.
        """
        ...

    @abstractmethod
    def get_r(self) -> List[Optional[float]]:
        """Calculates estimated reward per slot (without conditioning on impression). The length of the list is the maximum number of slots seen so far. Estimations per slot are calculated by CB estimator.

        Returns:
                Array of estimated reward values.
        """
        ...

    @abstractmethod
    def get_r_overall(self) -> Optional[float]:
        """Calculates estimated reward for sum of rewards over all slots.

        Returns:
                Estimated reward value
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
    def get_impression(self, alpha: float) -> List[Tuple[float, float]]:
        """Calculates impression probability per slot. The length of the list is the maximum number of slots seen so far.
        Args:
                alpha: alpha value

        Returns:
                Array of tuples (lower_bound / upper_bound)
        """
        ...

    @abstractmethod
    def get_r_given_impression(self, alpha: float) -> List[Tuple[float, float]]:
        """Calculates estimated reward per slot conditioned on impression. The length of the list is the maximum number of slots seen so far. Estimations per slot are calculated by CB estimator.
        Args:
                alpha: alpha value

        Returns:
                Array of tuples (lower_bound / upper_bound)
        """
        ...

    @abstractmethod
    def get_r(self, alpha: float) -> List[Tuple[float, float]]:
        """Calculates estimated reward per slot (without conditioning on impression). The length of the list is the maximum number of slots seen so far. Estimations per slot are calculated by CB estimator.
        Args:
                alpha: alpha value

        Returns:
                Array of tuples (lower_bound / upper_bound)
        """
        ...

    @abstractmethod
    def get_r_overall(self, alpha: float) -> Tuple[float, float]:
        """Calculates estimated reward for sum of rewards over all slots.
        Args:
                alpha: alpha value

        Returns:
                Tuple (lower_bound / upper_bound)
        """
        ...
