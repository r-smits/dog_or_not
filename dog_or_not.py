# Ramon Smits
# 2021-31-05
# dog_or_not ml classifier


from ml_hypothesis import Hypothesis
from ml_theta import Theta
from ml_predictions import Predictions

from ml_matrix import theta_transpose_x
from ml_matrix import calculate_gradient_descent


# Create hypothesis

print(
        """
--- The groundbreaking 'DOG_OR_NOT' ML classifier ---

# 5 training examples ->
#        A dog
#        A human
#        A flamingo
#        A dog (again)
#        A spider

# 4 features ->
# x0 -> decision boundary: choose wisely!
# x1 -> feature 1 : "number of legs"
# x2 -> feature 2 : "weight in kg"
# x3 -> feature 3 : "loudness on a scale from 1 - 10"
# x4 -> feature 4 : "friendliness to humans"

# y -> 
#       1 = Dog
#       0 = Not-Dog

---
""")

training_examples = [
        [0.5, 4, 30, 8, 9],                 # A dog
        [0.5, 2, 75, 6, 3],                 # A human
        [0.5, 1, 15, 10, 2],                # A flamingo
        [0.5, 4, 40, 9, 8],                 # A dog
        [0.5, 8, 0.140, 2, 1]               # A spider
        ]

exponents = [
        1,                      # Should be 1 scalar by default
        1,                      # x1 should be not be an exponent
        1,
        2,                      # exponents help with boundary condition
        2
        ]

expected_outcomes = [
        1,                      # Dog == Dog
        0,                      # Dog != Human
        0,                      # Dog != Flamingo
        1,                      # Dog == Dog
        0                       # Dog != Spider
        ]

h: Hypothesis = Hypothesis(training_examples, expected_outcomes, exponents)


thetas =    [[
            1,                  # This is regarding the decision boundary
            0.3,                # The number of weights should equal number of x's
            0.6,
            0.2,
            0.5
            ]]

t: Theta = Theta(thetas,0.01)

iterations: int = 20000
current: int = 1

while current <= iterations:
    
    p: Predictions = theta_transpose_x(t, h, sigmoid=True)
    t, predicted_vs_actual = calculate_gradient_descent(h, t, p)
    
    if current == 1:
        print(f"Very first prediction --- Iteration {current}")
        print(p)
        print(t)
        print(predicted_vs_actual)
    
    if current == iterations:
        print(f"Final prediction --- Iteration {current}")
        print(p)
        print(t)
        print(predicted_vs_actual)

    current = current + 1

