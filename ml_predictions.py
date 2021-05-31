from typing import List
from ml_sigmoid import calculate_sigmoids

class Predictions:
    P: List[List[float]] = []

    def __init__(self, _P: List[List[float]], sigmoid: bool=False):
        if sigmoid:
            for i in range(0, len(_P)):
                _P[i] = calculate_sigmoids(_P[i])
        self.P = _P
    
    def __str__(self):
        result: str = "--- Predictions ---"
        for row in self.P:
            result = f"{result}\n{str(row)}"
        return result

