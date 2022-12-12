""" Interface for implementation of conditional contextual bandits estimators """

from abc import ABC, abstractmethod
from typing import List, Optional


class Estimator(ABC):
	""" Interface for implementation of conditional contextual bandits estimators """

	@abstractmethod
	def add_example(self, p_log: List, r: List, p_pred: List) -> None:
		""" 
		Args:
			p_log: List of probability of the logging policy
			r: List of reward for choosing an action in the given context
			p_pred: List of predicted probability of making decision
		"""
		...

	@abstractmethod
	def get_impression(self) -> List[float]:
		""" Calculates impression probability for slot

		Returns:
			array of probability for every slot to be shown in decision
		"""
		...

	@abstractmethod
	def get_r_given_impression(self) -> List[float]:
		""" Calculates estimated reward for every slot given that the slot was shown
		
		Returns:
			array of estimated reward values
		"""
		...

	@abstractmethod
	def get_r(self) -> List[float]:
		""" Calculates estimated reward for every slot
		
		Returns:
			array of estimated reward values
		"""
		...

	@abstractmethod
	def get_r_overall(self) -> float:
		""" Calculates estimated reward for sum of rewards from all slots
		
		Returns:
			estimated value
		"""		
		...


class Interval(ABC):
	""" Interface for implementation of conditional contextual bandits estimators interval """

	@abstractmethod
	def add_example(self, p_log: List[float], r: List[float], p_pred: List[float], p_drop: float = 0, n_drop: Optional[int] = None) -> None:
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
	def get_impression(self, alpha) -> List[List[float]]:
		""" Calculates impression probability for slot

		Returns:
			array of [lower_bound, upper_bound] for every slot to be shown in decision
		"""
		...

	@abstractmethod
	def get_r_given_impression(self, alpha) -> List[List[float]]:
		""" Calculates estimated reward for every slot given that the slot was shown
		
		Returns:
			array of [lower_bound, upper_bound] of estimated reward values
		"""
		...

	@abstractmethod
	def get_r(self, alpha) -> List[List[float]]:
		""" Calculates estimated reward for every slot
		
		Returns:
			array of [lower_bound, upper_bound] of estimated reward values
		"""
		...

	@abstractmethod
	def get_r_overall(self, alpha) -> List[float]:
		""" Calculates estimated reward for sum of rewards from all slots
		
		Returns:
			[lower_bound, upper_bound] of estimated value
		"""		
		...
