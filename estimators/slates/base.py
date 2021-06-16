""" Interface for implementation of slates estimator """

from abc import ABC, abstractmethod
from typing import List

class Estimator(ABC):
	""" Interface for implementation of slates estimator """

	@abstractmethod
	def add_example(self, p_logs: List, r: float, p_preds: List, count: float) -> None:
		""" 
		Args:
			p_logs: List of probabilities of the logging policy
			r: reward for choosing an action in the given context
			p_preds: List of predicted probabilities of making decision
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
	""" Interface for implementation of slates estimator interval """

	@abstractmethod
	def add_example(self, p_logs: List, r: float, p_preds: List, count: float) -> None:
		""" 
		Args:
			p_logs: List of probabilities of the logging policy
			r: reward for choosing an action in the given context
			p_preds: List of predicted probabilities of making decision
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
