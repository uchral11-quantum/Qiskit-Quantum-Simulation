"""
Lagrange Multiplier Solver

Solves a system of nonlinear equations using Lagrange multipliers.
"""

import numpy as np
from scipy.optimize import fsolve

def equations(vars, x11, x12):
    lambda_11, lambda_12 = vars 
    epsilon_3 = -0.5 * (lambda_11 + np.sqrt(4 * lambda_12**2 + lambda_11**2))
    epsilon_4 = -0.5 * (lambda_11 - np.sqrt(4 * lambda_12**2 + lambda_11**2))
    k3 = -epsilon_3 / lambda_12
    k4 = -epsilon_4 / lambda_12
    Z = np.exp(epsilon_3) + np.exp(epsilon_4)
    a = (k3**2 / np.sqrt((k3**2 + 1) * (k3**2 + 1))) * np.exp(epsilon_3)
    b = (k4**2 / np.sqrt((k4**2 + 1) * (k4**2 + 1))) * np.exp(epsilon_4)

    eq1 = x11 - (1 / Z) * (a + b) 
    eq2 = x12 - (1 / Z) * (a / k3 + b / k4) 

    return [eq1, eq2]

x11_value = 0.483
x12_value = 0.499

initial_guess = [0.1, -0.2]  

solution = fsolve(equations, initial_guess, args=(x11_value, x12_value))

lambda_11_solution, lambda_12_solution = solution

print(f"λ11 = {lambda_11_solution}")
print(f"λ12 = {lambda_12_solution}")
