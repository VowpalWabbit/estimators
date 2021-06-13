""" Interface for implementation of slates estimator """

from abc import ABC, abstractmethod
from typing import List

class Estimator(ABC):
	""" Interface for implementation of slates estimator """

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
	def get_estimate(self) -> float:
		""" Calculates the selected estimator
		
		Returns:
			The estimator value
		"""
		...

class Interval(ABC):
	""" Interface for implementation of slates estimator interval """

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
	def get_interval(self, type: str, alpha: float) -> List:
		""" Calculates the CI
		Args:
			type: Specifies the interval type
			alpha: alpha value
		Returns:
			Returns the confidence interval as list[float]
		"""
		...