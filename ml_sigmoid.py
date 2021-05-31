from math import e
from typing import List


def calculate_sigmoid(z: float) -> float:
    res = 1 / (1 + e**-z)

    # This is normally where you use a boundary condition
    # However, I ran out of time.
    if res >= 0.5:
        return 1
    if res < 0.5:
        return 0


def calculate_sigmoids(Z: List[float]) -> List[float]:
    ress: List[float] = []
    for z in Z:
        ress.append(calculate_sigmoid(z))
    return ress

