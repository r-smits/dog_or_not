from typing import List
from ml_hypothesis import Hypothesis
from ml_theta import Theta, create_empty_theta
from ml_predictions import Predictions
from ml_sigmoid import calculate_sigmoids, calculate_sigmoid
from ml_cost import calculate_logistic_cost
from ml_matrix_base import Matrix, matrix_subtract_, matrix_add_, matrix_multiply_, transpose_, matrix_apply_, row_
import matplotlib.pyplot as plt

all_costs: List[float] = []


def plot_costs():
	x_labels = []
	print(f"plotting cost ...")
	for i in range(0, len(all_costs)):
		x_labels.append(i)
	plt.plot(x_labels, all_costs, 'o')
	plt.show()


def theta_transpose_x(theta: Matrix, hypothesis: Matrix, metrics=False) -> Predictions:
	# This function performs the following calculation:

	# Let Theta =   [t1]
	#               [t2]
	#               [t3]

	# Let X =       [x1]
	#               [x2]
	#               [x3]

	# Let Theta(transpose) =    [t1, t2, t3]

	# Theta(transpose) * X = 
	#
	# [t1, t2, t3] *    [x1] = [t1x1 + t2x2 + t3x3]
	#                   [x2]
	#                   [x3]
	
	m_predictions: Matrix = matrix_multiply_(theta, hypothesis)
	m_predictions: Matrix = matrix_apply_(m_predictions, calculate_sigmoid)
		
	# only to be used for diagnostics
	if metrics:
		cost = calculate_logistic_cost(m_predictions, hypothesis)
		print(f"[theta_transpose_x] cost = J(theta) = {cost}")
		all_costs.append(cost)
	
	return m_predictions


def calculate_gradient_descent(hypothesis: Hypothesis, theta: Matrix, predictions: Matrix, metrics=False) -> Theta:

	# This function calculates a gradient.
	# A gradient consists of slopes.
	# The slopes are retrieved from a derivative of a squared cost function.
	# cost(x) = (1/2n) * sum => (theta_transpose_x - actual_y)^2
	# d/dTheta = (1/n) * sum => (theta_transpose_x - actual_y) * x(i)(j)
	
	# vectorized to the extent that it can apply all calculations for an entire row of theta at the same time.	
	for i in range(0, theta.width):

		m_cost: Matrix = matrix_subtract_(predictions, hypothesis.vector_y())
		e_m_gradients: Matrix = matrix_multiply_(m_cost, transpose_(hypothesis))
		e_min_js: Matrix = matrix_multiply_(e_m_gradients, (-theta.alpha / hypothesis.width))
		new_theta: Matrix = matrix_add_(row_(theta, i), e_min_js)
		
		# only to be used for diagnostics
		if metrics:
			print(theta)
			print(new_theta)
		
		theta.points[i] = new_theta.points[0]
		
	return theta

