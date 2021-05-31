from typing import List
from ml_min_max import min_max_normalization


class Hypothesis:
    X: List[List[float]] = []
    Y: List[float] = []
    exp: List[float] = []

    def __init__(self, _X: List[List[float]], _Y: List[float], exponents: List[int]):
        self.X = min_max_normalization(_X)
        self.Y = _Y
        self.exp = exponents
        
    def __str__(self):
        result: str = "--- Hypothesis ---"
        for i in range(0, len(self.X)):
            result = f"{result}\n{str(self.X[i])} -> {self.Y[i]}"
        return result

