from typing import List
from ml_matrix_base import Matrix


class Hypothesis(Matrix):
	exp: List[float] = []
	y: List[float] = []

	def __init__(self,
														_X: List[List[float]],
														_Y: List[float],
														_exp: List[int]=None):
		super().__init__(_X, "Hypothesis")
		self.y = _Y
		self.exp = _exp

	def set_exp(self, _exp: List[int]):
		self.exp = _exp

	def get_y(self, i: int) -> float:
		return self.y[i]
	
	def vector_y(self) -> Matrix:
		return Matrix([self.y], f"y_actual from {self.name}")

