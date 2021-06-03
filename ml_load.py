from io import TextIOWrapper
from typing import List
from ml_hypothesis import Hypothesis
from ml_feature_scaling import standardize_values


def data_not_empty_and_has_same_length(buffer: TextIOWrapper) -> bool:
	length: int = -1
	for line in buffer:
		if length == -1:
			length = len(line)
		if len(line) != length:
			return False
		return False if length <= 1 else True


# Import functionality does not expect exponents!
# Set the exponent values later in the code.
def import_csv_file(path: str) -> Hypothesis:
	file: TextIOWrapper = open(path, 'r')
	x_values: List[List[float]] = []
	y_values: List[float] = []

	if not data_not_empty_and_has_same_length(file):
		print(
			f"[import_csv_file] File must contain at least 2 values per training example."
		)
		print(f"Every training example must be of the same length.")
		raise Exception()
	file.seek(0)

	for line in file:
		values: List[float] = line.replace('\n', '').replace(' ', '').split(',')
		try:
			values = list(map(lambda val: float(val), values))
			x_values.append(values[:len(values) - 1])
			y_values.append(values[len(values) - 1])
		except Exception as e:
			print(f"[import_csv_file] Could not cast '{values}' to float.")
	return Hypothesis(x_values, y_values)


# Don't forget : set your exponents later in the code.
def import_csv_file_sigmoid(path: str) -> List[Hypothesis]:
	file: TextIOWrapper = open(path, 'r')
	x_values: List[List[float]] = []
	y_values: List[float] = []

	if not data_not_empty_and_has_same_length(file):
		print(
			f"[import_csv_file] \nFile must contain at least 2 values per training example."
		)
		print(f"Every training example must be of the same length.")
		raise Exception()
	file.seek(0)

	for line in file:
		values: List[float] = line.replace('\n', '').replace(' ', '').split(',')
		try:
			values = list(map(lambda val: float(val), values))
			x_values.append(values[:len(values) - 1])
			y_values.append(values[len(values) - 1])
		except Exception as e:
			print(f"[import_csv_file] Could not cast '{values}' to float.")

	y_max: int = int(max(y_values))
	sigmoid_y_values: List[List[float]] = []
	for i in range(1, y_max + 1):
		sigmoid_y_values.append([1 if val == i else 0 for val in y_values])

	hypotheses: List[Hypothesis] = []
	for sigmoid_y in sigmoid_y_values:
		hypothesis: Hypothesis = Hypothesis(x_values, sigmoid_y)
		hypothesis: Hypothesis = Hypothesis(standardize_values(hypothesis).points, sigmoid_y)
		hypotheses.append(hypothesis)
	return hypotheses

