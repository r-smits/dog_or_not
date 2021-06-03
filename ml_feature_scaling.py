from typing import List, Dict
from ml_matrix_base import Matrix, matrix_subtract_, matrix_multiply_
from math import sqrt


# Mean, the average of the values
def calculate_mean(matrix: Matrix) -> float:
		sum: float = matrix.total_summation()
		return sum / (matrix.width * matrix.height)


# The standard deviation as regarded on a bell curve
def calculate_standard_deviation(matrix: Matrix) -> float:
	sum: float = matrix.total_summation()
	sq_sum: float = sum ** 2
	av_sq: float = sq_sum / (matrix.width * matrix.height)
	
	m_points = matrix.points
	sep_sum: float = 0
	for i in range(0, matrix.width):
		for j in range(0, matrix.height):
			av_sq += m_points[i][j] ** 2
	
	diff_sum = av_sq - sep_sum
	variance: float = diff_sum / (matrix.width * matrix.height - 1)
	standard_deviation: float = sqrt(variance)
	return standard_deviation


def standardize_values(matrix: Matrix) -> Matrix:
	mean: float = calculate_mean(matrix)
	standard_deviation = calculate_standard_deviation(matrix)
	matrix_standardized: Matrix = matrix_multiply_(matrix_subtract_(matrix, mean), (1 / standard_deviation))
	print(matrix_standardized)
	return matrix_standardized



