# Ramon Smits
# 2021-03-06
# dog_or_not ml classifier


import time
from typing import List
from ml_hypothesis import Hypothesis
from ml_theta import Theta
from ml_predictions import Predictions
from ml_load import import_csv_file_sigmoid

from ml_gradient import theta_transpose_x
from ml_gradient import calculate_gradient_descent
from ml_gradient import plot_costs


print("""
--- The groundbreaking 'DOG_OR_NOT' ML classifier ---

# 5 training examples ->
#        A dog
#        A human
#        A flamingo
#        A dog (again)
#        A spider

# 4 features ->
# x0 -> decision boundary: choose wisely!
# x1 -> feature 1 : "number of legs"
# x2 -> feature 2 : "weight in kg"
# x3 -> feature 3 : "loudness on a scale from 1 - 10"
# x4 -> feature 4 : "friendliness to humans"

# y -> 
#       1 = Dog
#       0 = Not-Dog

---
""")


# The below function separates training data in a training set, and an expected binary set of outcomes.
# If there are multiple categories, this means multiple hypothesis are created with varying binary outcomes. E.g.

#training_examples = [
#	[1, 4, 30, 8, 9],  			# A dog
#	[1, 2, 75, 6, 3],  			# A human
#	[1, 1, 15, 10, 2],  		# A flamingo
#	[1, 4, 40, 9, 8],  			# A dog
#	[1, 8, 0.140, 2, 1]  		# A spider
#]
#
#expected_outcomes = [
#	1,  # Dog == Dog
#	0,  # Dog != Human
#	0,  # Dog != Flamingo
#	1,  # Dog == Dog
#	0  # Dog != Spider
#]
hypotheses: List[Hypothesis] = import_csv_file_sigmoid("training_data.csv")
exponents = [1, 1, 1, 1, 1]
for hypothesis in hypotheses:
	hypothesis.set_exp(exponents)


thetas = [
	[0, 0, 0, 0, 0]
	]

alphas = [0.5]
epochs: int = 4000

trained_classifiers: List[Theta] = []

for hypothesis in hypotheses:
	for alpha in alphas:
		current: int = 1

		theta: Theta = Theta(thetas, alpha)
		
		while current <= epochs:

			predictions: Predictions = theta_transpose_x(theta, hypothesis, metrics=True)
			theta: Theta = calculate_gradient_descent(hypothesis, theta, predictions)
			current = current + 1
		
		print(theta)
		trained_classifiers.append(theta)
	break


plot_costs()
		

