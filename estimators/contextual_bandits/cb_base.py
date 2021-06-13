""" Interface for implementation of contextual bandit estimators """

from abc import ABC, abstractmethod
from typing import List

class Estimator(ABC):
	""" Interface for implementation of contextual bandit estimators """

	@abstractmethod
	def add_example(self, p_log: float, r: float, p_pred: float, count: float) -> None:
		""" 
		Args:
			p_log: probability of the logging policy
			r: reward for choosing an action in the given context
			p_pred: predicted probability of making decision
			count: weight
		"""
		...

	@abstractmethod
	def get(self) -> float:
		""" Calculates the selected estimator
		
		Returns:
			The estimator value
		"""
		...

class Interval(ABC):
	""" Interface for implementation of contextual bandit estimators interval """

	@abstractmethod
	def add_example(self, p_log: float, r: float, p_pred: float, count: float) -> None:
		""" 
		Args:
			p_log: probability of the logging policy
			r: reward for choosing an action in the given context
			p_pred: predicted probability of making decision
			count: weight
		"""
		...

	@abstractmethod
	def get(self, alpha: float) -> List:
		""" Calculates the CI
		Args:
			alpha: alpha value
		Returns:
			Returns the confidence interval as list[float]
		"""
		...
