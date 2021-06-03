from typing import List
from math import log
from ml_matrix_base import Matrix
from ml_hypothesis import Hypothesis


def calculate_cost(y_predicted: List[float], y_actual: List[float]) -> float:

	sum_norm_squared_cost: float = 0
	for i in range(0, len(y_predicted)):
		squared_cost: float = (y_predicted[i] - y_actual[i])**2
		norm_squared_cost: float = 0.5 * squared_cost
		sum_norm_squared_cost = sum_norm_squared_cost + norm_squared_cost
	av_sum_norm_squared_cost: float = (
		1 / len(y_predicted)) * sum_norm_squared_cost
	return av_sum_norm_squared_cost


def calculate_logistic_cost(predictions: Matrix, actual: Hypothesis) -> float:
	sum_cost: float = 0
	for i in range(0, predictions.width):
		for j in range(0, predictions.height):
			y_predicted: float = predictions.get(i, j)
			y_actual: float = actual.get_y(j)
			cost = -y_actual * log(y_predicted) - (1 - y_actual) * log(1 - y_predicted)
			sum_cost += cost
	av_cost: float = (1 / predictions.height) * sum_cost
	return av_cost

