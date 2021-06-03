from typing import List


class Matrix:

	points: List[List[float]]
	name: str
	width: int
	height: int

	def __init__(self, _points: List[List[float]], _name: str):
		self.points = _points
		self.name = _name
		self.width = len(_points)
		self.height = len(_points[0])

	def get(self, i: int, j: int) -> float:
		return self.points[i][j]
		
	def summation(self, i: int) -> float:
		return sum(self.points[i])
		
	def total_summation(self) -> float:
		total: float = sum([self.summation(i) for i in range(0, self.width)])
		return total

	def __str__(self) -> str:
		result: str = f"\n--- {self.name} ---> \n"
		for row in self.points:
			result += "["
			for val in row:
				result += '%10f' % val
			result += '%5s' % " ] \n"
		result += f"> Size: ({self.width}x{self.height})\n"
		return result


def vectorize_(main: Matrix, i: int) -> Matrix:
		m_points: List[float] = [row[i] for row in main.points]
		return Matrix([m_points], f"{main.name}(i)({i})")


def row_(main: Matrix, i: int) -> Matrix:
	return Matrix([main.points[i]], f"{main.name}({i})(j)")


def transpose_(main: Matrix) -> Matrix:
	m_points: List[List[float]] = []
	for j in range(0, main.height):
		m_points.append([])
		for i in range(0, main.width):
			m_points[j].append(main.get(i, j))
	return Matrix(m_points, f"transpose({main.name})")
		
	
def matrix_subtract_(main: Matrix, other) -> Matrix:
	m_points: List[List[float]] = []
	if isinstance(other, float) or isinstance(other, int):
		for i in range(0, main.width):
			m_points.append([])
			for j in range(0, main.height):
				m_points[i].append(main.get(i, j) - other)
		return Matrix(m_points, f"({main.name} - {other})")
		
	if isinstance(other, Matrix):
		for i in range(0, main.width):
			m_points.append([])
			for j in range(0, main.height):
				m_points[i].append(main.get(i, j) - other.get(i, j))
		return Matrix(m_points, f"({main.name} - {other.name})")
	

def matrix_add_(main: Matrix, other) -> Matrix:
		m_points: List[List[float]] = []
		if isinstance(other, float) or isinstance(other, int):
			for i in range(0, main.width):
				m_points.append([])
				for j in range(0, main.height):
					m_points[i].append(main.get(i, j) + other)
			return Matrix(m_points, f"({main.name} + {other})")
		
		if isinstance(other, Matrix):
			for i in range(0, main.width):
				m_points.append([])
				for j in range(0, main.height):
					m_points[i].append(main.get(i, j) + other.get(i, j))
			return Matrix(m_points, f"({main.name} + {other.name})")
		

def matrix_multiply_(main: Matrix, other):
		if isinstance(other, float) or isinstance(other, int):
			m_points: List[List[float]] = []
			for i in range(0, main.width):
				m_points.append([])
				for j in range(0, main.height):
					m_points[i].append(main.get(i, j) * other)
			return Matrix(m_points, f"({main.name} x {other})")
		
		if isinstance(other, Matrix):
			m_points: List[List[float]] = []
			for i in range(0, main.width):
				m_points.append([])
				for ii in range(0, other.width):
					m_points[i].append(0)
					for j in range(0, other.height):
						m_points[i][ii] += (main.get(i, j) * other.get(ii, j))
			return Matrix(m_points, f"({main.name} x {other.name})")


def matrix_apply_(main: Matrix, func) -> Matrix:
	# Make sure that the function only takes one argument, which should be a float.
	m_points: List[List[float]] = []
	for i in range(0, main.width):
		m_points.append([])
		for j in range(0, main.height):
			m_points[i].append(func(main.get(i, j)))
	return Matrix(m_points, f"{func.__name__}({main.name})")
	

