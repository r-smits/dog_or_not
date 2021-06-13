import matplotlib.pyplot as plt
from typing import List
from core.ml_cost import calculate_logistic_cost
from dto.ml_matrix_base import Matrix
from dto.ml_predictions import Predictions


class Plot:
	
	all_costs: List[float]
	
	def __init__(self):
		self.all_costs = []
		axes = plt.gca()
	
	def calculate_cost(self, predictions: Predictions, y_actual: Matrix) -> Matrix:
		cost: Matrix = calculate_logistic_cost(predictions, y_actual)
		self.all_costs.extend(cost.points)
		print(f"[theta_transpose_x] cost = J(theta) = {cost}")
		
	def add(self, matrix: Matrix):
		self.all_costs.extend(matrix.points)
	
	def show(self):
		x_labels = []
		print(f"plotting cost ...")
		for i in range(0, len(self.all_costs)):
			x_labels.append(i)
		plt.plot(x_labels, self.all_costs, 'o')
		plt.show()

