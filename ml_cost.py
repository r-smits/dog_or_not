from typing import List
from math import log


def calculate_cost(y_predicted: List[float], y_actual: List[float]) -> float:
    
    sum_norm_squared_cost: float = 0
    for i in range(0, len(y_predicted):
        squared_cost: float = (y_predicted[i] - y_actual[i]) ** 2
        norm_squared_cost: float = 0.5 * squared_cost
        sum_norm_squared_cost = sum_norm_squared_cost + norm_squared_cost
    av_sum_norm_squared_cost: float = (1 / len(y_predicted) * sum_norm_squared_cost
    return av_sum_squared_cost    


def calculate_logistic_cost(y_predicted: List[float], y_actual: List[float]) -> float:   
    sum_log_cost: float = 0 
    for i in range(0, len(y_predicted):
        result = -log(y_predicted[i]) if int(y_actual[i]) == 1 else -log(1 - y_predicted[i])
        sum_log_cost = sum_log_cost + result
    av_sum_log_cost: float = (1 / len(y_predicted) * sum_log_cost
    return av_sum_log_cost

