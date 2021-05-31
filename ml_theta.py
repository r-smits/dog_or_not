from typing import List


class Theta:
    T: List[List[float]] = []
    alpha: float = 0.005

    def __init__(self, _T: List[List[float]], alpha: float):
        self.T = _T
        self.alpha = alpha

    def __str__(self):
        result: str = "--- Theta ---"
        for row in self.T:
            result = f"{result}\n{str(row)}"
        return result

