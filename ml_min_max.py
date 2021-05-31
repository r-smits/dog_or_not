from typing import List


def min_max_normalization(values: List[List[float]]) -> List[List[float]]:
    all_values = []
    [all_values.extend(value_list) for value_list in values]

    minimum = min(all_values)
    maximum = max(all_values)
    diff = maximum - minimum
    
    for i in range(0, len(values)):
        for j in range(0, len(values[i])):
            values[i][j] = (values[i][j] - minimum) / diff
    
    return values
