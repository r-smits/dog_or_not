from typing import List
from ml_hypothesis import Hypothesis
from ml_theta import Theta
from ml_predictions import Predictions


def theta_transpose_x(t: Theta, h: Hypothesis, sigmoid=False) -> Predictions:
    
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
    # [t1, t2, t3] *    [x1] = [t1x1 + t2x2, t3x3]
    #                   [x2]
    #                   [x3]

    result: List[List[float]] = []
    for i in range(0, len(t.T)):
        row_predicted: List[float] = []
        for j in range(0, len(h.X)):
            calculation: str = ""
            y_predicted: float = 0
            for k in range(0, len(t.T[i])):
                y_predicted = y_predicted + (t.T[i][k] * h.X[j][k] ** h.exp[k])
                calculation = calculation + f" + {t.T[i][k]} * x ^ {h.exp[k]}"
            # print(f"calculation: {calculation} = {y_predicted}")
            row_predicted.append(y_predicted)
        result.append(row_predicted)
    p: Predictions = Predictions(result, sigmoid)
    return p


def calculate_gradient_descent(h: Hypothesis, t: Theta, p: Predictions) -> (Theta, str):
    
    # This function calculates a gradient.
    # A gradient consists of slopes.
    # The slopes are retrieved from a derivative of a squared cost function.
    # cost(x) = (1/2n) * sum => (theta_transpose_x - actual_y)^2
    # d/dTheta = (1/n) * sum => (theta_transpose_x - actual_y) * x(i)(j)

    for i in range(0, len(t.T)):
        for k in range(0, len(t.T[i])):
            predicted_vs_actual: str = ""
            
            gradient: List[float] = []
            for j in range(0, len(h.X)):
                
                y_predicted = p.P[i][j]
                y_actual = h.Y[j]
                
                slope: float = (y_predicted - y_actual) * (h.X[j][k] ** h.exp[k])
                gradient.append(slope)

                predicted_vs_actual = predicted_vs_actual + f"predicted: {y_predicted}, actual: {y_actual}\n"
            
            # alpha = the learning rate
            # sum(gradient) / len(gradient) = the average slope that needs to be descended against

            min_j: float = -t.alpha * (sum(gradient) / len(gradient))
            t.T[i][k] = t.T[i][k] + min_j
            
            # print(f"gradient: {gradient}, average: {sum(gradient) / len(gradient)}, alpha: {t.alpha}")
             
    return (t, predicted_vs_actual)


