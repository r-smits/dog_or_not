from typing import List
from ml_hypothesis import Hypothesis
from ml_theta import Theta
from ml_predictions import Predictions
from ml_sigmoid import calculate_sigmoid


def predict(training_example, weights) -> float:
	prediction = weights[0]
	for i in range(1, len(training_example)):
		prediction += weights[i] * training_example[i]
	prediction = calculate_sigmoid(prediction)
	return prediction
		
	
def calculate_stochastic_gradient_descent(h: Hypothesis, t: Theta):
	
	sum_error: float = 0
		
	for i in range(0, len(h.X)):
		for j in range(0, len(t.T)):
			
			training_example = h.X[i]
			weights = t.T[j]
			actual_outcome = h.Y[i]
			predicted_outcome = predict(training_example, weights)
			print(f"predicted outcome: {predicted_outcome}, actual outcome: {actual_outcome}")
			
			assert len(training_example) == len(weights)
			
			error = actual_outcome - predicted_outcome
			sum_error += error ** 2
			
			# For theta 0
			weights[0] + t.alpha * error * predicted_outcome * (1 - predicted_outcome)
			# For all other theta's'
			for x in range(1, len(training_example)):
				weights[x] = weights[x] + t.alpha * error * predicted_outcome * (1 - predicted_outcome) * training_example[x]
			
			print(f"learning rate: {t.alpha}, error={sum_error}")
			
			t.T[j] = weights
		
		print(t)
