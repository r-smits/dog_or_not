from math import e
from typing import List
from ml_matrix_base import Matrix, matrix_multiply_


def calculate_sigmoid(z: float) -> float:
    res = 1 / (1 + (e**-z))

    # This is normally where you use a boundary condition
    # However, I ran out of time.
    return res


def calculate_sigmoids(Z: List[float]) -> List[float]:
    ress: List[float] = []
    for z in Z:
        ress.append(calculate_sigmoid(z))
    return ress

def apply_sigmoid(matrix: Matrix) -> Matrix:
	
	matrix_multiply_(matrix, 1 / ())
	


