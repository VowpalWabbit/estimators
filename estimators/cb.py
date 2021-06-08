""" Interface for implementation of Contextual Bandit estimators """

from abc import ABC, abstractmethod
from typing import List, Dict

class ContextualBandits(ABC):
	""" Interface for implementation of Contextual Bandit estimators """

	@abstractmethod
	def add_example(self, p_log: float, r: float, p_pred: float, count: int) -> None:
		""" 

		Args:
			p_log: probability
			r: reward value
			p_pred:
			count:

		"""
		...

	@abstractmethod
	def get_estimate(self, info: Dict) -> float:
		""" Calculates the selected estimator

		Args:
			type: specifies the estimator to be used

		Returns:
			The estimator value
		"""
		...

	@abstractmethod
	def get_interval(self, info: Dict, alpha: float) -> List:
		""" Calculates the CI

		Args:
			type: Specifies the interval type
			alpha: alpha value

		Returns:
			Returns the confidence interval as list[float]
		"""
		...