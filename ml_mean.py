from typing import List


# Mean, the average of the values
def mean(values: List[float]) -> float:
    return sum(values) / len(values)


# Variance, sum of squared difference between values and the average value
def variance(values: List[float]) -> float:
    m: float = mean(values)
    return sum([(x-m)**2 for x in values])


def coveriance(x: List[float], y: List[float]) -> float:
    covar: float = 0
    x_mean: float = mean(x)
    y_mean: float = mean(y)
    for i in range(0, len(x)):
    [covar = covar + (x[i] - x_mean) * (y[i] - y_mean) for i in range(0, len(x))]
    return covar