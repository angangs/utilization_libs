from __future__ import print_function
from ortools.linear_solver import pywraplp
import numpy as np


class Optimizer:
    def __init__(self):
        self.solver = None
        self.status = None

    def get_solver(self, mode):
        if mode == "linear":
            self.solver = pywraplp.Solver.CreateSolver('GLOP')
        elif mode == "mip":
            self.solver = pywraplp.Solver.CreateSolver('SCIP')
        else:
            pass

    def solve(self, obj, seconds_to_wait=60):
        self.solver.Minimize(obj)
        if seconds_to_wait is not None:
            self.solver.set_time_limit(seconds_to_wait * 1000)
        self.status = self.solver.Solve(obj)

    def set_linear_variables(self, var_name, lower_limit, uppper_limit):
        self.solver.NumVar(lower_limit, uppper_limit, var_name)

    def set_int_variables(self, var_name, lower_limit,  uppper_limit):
        self.solver.IntVar(lower_limit, uppper_limit, var_name)

    def set_abs_constraint(self, variable, const_var):
        # Constraint: |X| < K
        self.solver.Add(variable >= const_var)
        self.solver.Add(variable <= const_var)

    def set_equality_constraint(self, left, right):
        self.solver.Add(left == right)

    def set_inequality_constraint(self, left, right):
        self.solver.Add(left <= right)

    def get_solution(self, variable):
        if self.status == pywraplp.Solver.OPTIMAL or self.status == pywraplp.Solver.FEASIBLE:
            solution_list = []
            for i in range(variable.shape[0]):
                for j in range(variable.shape[1]):
                    solution_list.append(variable.solution_value(variable[i, j]))
            return np.reshape(solution_list, (variable.shape[0], variable.shape[1]))
        else:
            return None
