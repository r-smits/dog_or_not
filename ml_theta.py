from typing import List
from ml_matrix_base import Matrix


class Theta(Matrix):
    alpha: float
    lambd: float

    def __init__(self, _T: List[List[float]], alpha: float):
        super().__init__(_T, "Theta")
        self.alpha = alpha
        self.lambd = 0.005


def create_empty_theta(t: Theta) -> Theta:
    empty_theta_vals: List[List[float]] = []
    for i in range(0, t.l):
        empty_theta_arr: List[float] = [0] * t.w
        empty_theta_vals.append(empty_theta_arr)
    return Theta(empty_theta_vals)

