""" Interface for implementation of contextual bandit estimators """

from abc import ABC, abstractmethod
from typing import List, Optional


class Estimator(ABC):
	""" Interface for implementation of contextual bandit estimators """

	@abstractmethod
	def add_example(self, p_log: float, r: float, p_pred: float) -> None:
		""" 
		Args:
			p_log: probability of the logging policy
			r: reward for choosing an action in the given context
			p_pred: predicted probability of making decision
		"""
		...

	@abstractmethod
	def get(self) -> Optional[float]:
		""" Calculates the selected estimator
		
		Returns:
			The estimator value
		"""
		...


class Interval(ABC):
	""" Interface for implementation of contextual bandit estimators interval """

	@abstractmethod
	def add_example(self, p_log: float, r: float, p_pred: float, p_drop: float = 0, n_drop: Optional[int] = None) -> None:
		""" 
		Args:
			p_log: probability of the logging policy
			r: reward for choosing an action in the given context
			p_pred: predicted probability of making decision
			p_drop: probability for event to be dropped
			n_drop: amount of dropped events between current and previous ones (populated as p_drop/(1-p_drop) if None)
		"""
		...

	@abstractmethod
	def get(self, alpha: float) -> List[Optional[float]]:
		""" Calculates the CI
		Args:
			alpha: alpha value
		Returns:
			Returns the confidence interval as list[float]
		"""
		...
