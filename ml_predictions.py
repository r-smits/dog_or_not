from typing import List
from ml_sigmoid import calculate_sigmoids
from ml_matrix_base import Matrix


class Predictions(Matrix):
	def __init__(self, _P: List[List[float]]):
		super().__init__(_P, "Predictions")

