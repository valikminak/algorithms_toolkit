from typing import List, Dict, Tuple, Set, Optional, Any, Union
import numpy as np
import collections
import math


class SimplexSolver:
    """
    Simplex algorithm for solving linear programming problems.

    This implementation solves problems in the form:
    Maximize c^T x
    Subject to Ax <= b
    and x >= 0
    """

    def __init__(self, c: List[float], A: List[List[float]], b: List[float]):
        """
        Initialize a linear programming problem in standard form.

        Args:
            c: Coefficients of the objective function
            A: Constraint coefficients matrix
            b: Constraint right-hand side values
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)

        # Check dimensions
        if len(c) != self.A.shape[1]:
            raise ValueError("Dimension mismatch: c has length {}, but A has {} columns".format(
                len(c), self.A.shape[1]))

        if len(b) != self.A.shape[0]:
            raise ValueError("Dimension mismatch: b has length {}, but A has {} rows".format(
                len(b), self.A.shape[0]))

        # Check for negative values in b
        if np.any(self.b < 0):
            raise ValueError("Negative values in b are not supported in this implementation")

    def solve(self) -> Tuple[Optional[List[float]], Optional[float]]:
        """
        Solve the linear programming problem using the simplex algorithm.

        Returns:
            Tuple of (optimal_solution, optimal_value) if a solution exists,
            or (None, None) if the problem is unbounded
        """
        # Create the initial tableau
        tableau = self._create_initial_tableau()

        # Run the simplex algorithm
        tableau = self._simplex(tableau)

        # Check if the problem is unbounded
        if tableau is None:
            return None, None

        # Extract the solution
        solution = self._extract_solution(tableau)

        # Calculate the optimal value
        optimal_value = np.dot(self.c, solution)

        return solution.tolist(), float(optimal_value)

    def _create_initial_tableau(self) -> np.ndarray:
        """
        Create the initial simplex tableau.

        Returns:
            The initial tableau as a numpy array
        """
        m, n = self.A.shape

        # Create the tableau: [A | b]
        #                    [-c | 0]
        tableau = np.zeros((m + 1, n + 1))
        tableau[:m, :n] = self.A
        tableau[:m, n] = self.b
        tableau[m, :n] = -self.c

        return tableau

    def _simplex(self, tableau: np.ndarray) -> Optional[np.ndarray]:
        """
        Run the simplex algorithm on the tableau.

        Args:
            tableau: The initial tableau

        Returns:
            The final tableau, or None if the problem is unbounded
        """
        m, n = tableau.shape
        m -= 1  # Number of constraints
        n -= 1  # Number of variables

        # Keep track of basic variables
        basic_vars = [-1] * m

        # Simplex iterations
        while True:
            # Find the entering variable (most negative coefficient in objective row)
            z = tableau[m, :n]
            entering_col = np.argmin(z)

            # Check optimality
            if z[entering_col] >= -1e-10:
                break

            # Find the leaving variable (minimum ratio test)
            ratios = []
            for i in range(m):
                if tableau[i, entering_col] <= 1e-10:
                    ratios.append(float('inf'))
                else:
                    ratios.append(tableau[i, n] / tableau[i, entering_col])

            # Check if the problem is unbounded
            if all(r == float('inf') for r in ratios):
                return None

            leaving_row = np.argmin(ratios)
            basic_vars[leaving_row] = entering_col

            # Pivot
            tableau = self._pivot(tableau, leaving_row, entering_col)

        return tableau

    def _pivot(self, tableau: np.ndarray, leaving_row: int, entering_col: int) -> np.ndarray:
        """
        Perform a pivot operation on the tableau.

        Args:
            tableau: The current tableau
            leaving_row: Index of the leaving row
            entering_col: Index of the entering column

        Returns:
            The updated tableau
        """
        m, n = tableau.shape

        # Copy the tableau
        new_tableau = tableau.copy()

        # Normalize the pivot row
        pivot = tableau[leaving_row, entering_col]
        new_tableau[leaving_row, :] = tableau[leaving_row, :] / pivot

        # Update other rows
        for i in range(m):
            if i != leaving_row:
                factor = tableau[i, entering_col]
                new_tableau[i, :] = tableau[i, :] - factor * new_tableau[leaving_row, :]

        return new_tableau

    def _extract_solution(self, tableau: np.ndarray) -> np.ndarray:
        """
        Extract the solution from the final tableau.

        Args:
            tableau: The final tableau

        Returns:
            The optimal solution vector
        """
        m, n = tableau.shape
        m -= 1  # Number of constraints
        n -= 1  # Number of variables

        # Initialize solution vector
        solution = np.zeros(n)

        # Find basic variables
        for i in range(m):
            # Find the basic variable in this row
            basic_col = -1
            for j in range(n):
                if abs(tableau[i, j] - 1.0) < 1e-10 and all(abs(tableau[k, j]) < 1e-10 for k in range(m) if k != i):
                    basic_col = j
                    break

            if basic_col != -1:
                solution[basic_col] = tableau[i, n]

        return solution


class InteriorPointSolver:
    """
    Interior point method (path-following) for solving linear programming problems.

    This implementation solves problems in the form:
    Maximize c^T x
    Subject to Ax <= b
    and x >= 0
    """

    def __init__(self, c: List[float], A: List[List[float]], b: List[float], epsilon: float = 1e-6,
                 max_iter: int = 100):
        """
        Initialize a linear programming problem in standard form.

        Args:
            c: Coefficients of the objective function
            A: Constraint coefficients matrix
            b: Constraint right-hand side values
            epsilon: Convergence tolerance
            max_iter: Maximum number of iterations
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.epsilon = epsilon
        self.max_iter = max_iter

        # Check dimensions
        if len(c) != self.A.shape[1]:
            raise ValueError("Dimension mismatch: c has length {}, but A has {} columns".format(
                len(c), self.A.shape[1]))

        if len(b) != self.A.shape[0]:
            raise ValueError("Dimension mismatch: b has length {}, but A has {} rows".format(
                len(b), self.A.shape[0]))

    def solve(self) -> Tuple[Optional[List[float]], Optional[float]]:
        """
        Solve the linear programming problem using the interior point method.

        Returns:
            Tuple of (optimal_solution, optimal_value) if a solution exists,
            or (None, None) if the problem is infeasible or unbounded
        """
        # Convert to standard form with slack variables
        n, m = self.A.shape
        c_bar = np.concatenate([self.c, np.zeros(m)])
        A_bar = np.hstack([self.A, np.eye(m)])

        # Initialize at a feasible interior point
        x = np.ones(n + m)
        s = np.ones(m)
        y = np.zeros(m)

        # Path-following algorithm
        for _ in range(self.max_iter):
            # Compute duality gap
            gap = np.dot(s, x[n:]) / m

            # Check convergence
            if gap < self.epsilon:
                break

            # Compute the step
            step = self._compute_step(A_bar, c_bar, x, y, s, gap)

            # Line search to ensure s > 0 and x[n:] > 0
            alpha_p, alpha_d = self._line_search(x, s, step)

            # Update variables
            x = x + alpha_p * step[:n + m]
            y = y + alpha_d * step[n + m:n + 2 * m]
            s = s + alpha_d * step[n + 2 * m:]

        # Extract solution for original variables
        solution = x[:n]

        # Calculate the optimal value
        optimal_value = np.dot(self.c, solution)

        return solution.tolist(), float(optimal_value)

        # advanced/linear_programming.py (continued)

    def _compute_step(self, A: np.ndarray, c: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray,
                      gap: float) -> np.ndarray:
        """
        Compute the step for the interior point method.

        Args:
            A: Constraint matrix in standard form
            c: Objective coefficients in standard form
            x: Current primal variables
            y: Current dual variables
            s: Current slack variables
            gap: Current duality gap

        Returns:
            Step direction for primal and dual variables
        """
        n, m = A.shape
        n -= m  # Number of original variables

        # Target gap
        sigma = 0.1
        mu = sigma * gap

        # Extract slack variables
        x_slack = x[n:]

        # Compute the residuals
        r_p = np.dot(A, x) - self.b
        r_d = np.dot(A.T, y) + np.concatenate([np.zeros(n), s]) - c
        r_c = x_slack * s - mu

        # Form the augmented system
        D = np.diag(s / x_slack)
        M = np.dot(A[:, n:], np.dot(np.diag(1.0 / s), np.diag(x_slack)))

        # Solve for dy
        rhs = r_p - np.dot(A[:, n:], r_c / s)
        dy = np.linalg.solve(M, rhs)

        # Compute ds and dx
        ds = np.dot(A[:, n:].T, dy) - r_d[n:]
        dx_slack = (mu - x_slack * s - x_slack * ds) / s
        dx = np.concatenate([np.zeros(n), dx_slack])

        # For original variables, solve using dual residual
        dx[:n] = np.linalg.solve(A[:, :n].T, -r_d[:n] - np.dot(A[:, n:].T, dy))

        return np.concatenate([dx, dy, ds])

    def _line_search(self, x: np.ndarray, s: np.ndarray, step: np.ndarray) -> Tuple[float, float]:
        """
        Perform a line search to ensure x and s remain positive.

        Args:
            x: Current primal variables
            s: Current slack variables
            step: Step direction

        Returns:
            Tuple of (primal_step_size, dual_step_size)
        """
        n = len(x) - len(s)
        dx = step[:len(x)]
        ds = step[-len(s):]

        # Compute maximum step sizes
        alpha_p = 1.0
        for i in range(n, len(x)):
            if dx[i] < 0:
                alpha_p = min(alpha_p, -0.99 * x[i] / dx[i])

        alpha_d = 1.0
        for i in range(len(s)):
            if ds[i] < 0:
                alpha_d = min(alpha_d, -0.99 * s[i] / ds[i])

        return alpha_p, alpha_d

def solve_lp(c: List[float], A: List[List[float]], b: List[float], method: str = 'simplex') -> Tuple[
    Optional[List[float]], Optional[float]]:
    """
    Solve a linear programming problem.

    Args:
        c: Coefficients of the objective function (maximize c^T x)
        A: Constraint coefficients matrix (Ax <= b)
        b: Constraint right-hand side values
        method: Solution method ('simplex' or 'interior-point')

    Returns:
        Tuple of (optimal_solution, optimal_value) if a solution exists,
        or (None, None) if the problem is infeasible or unbounded
    """
    if method == 'simplex':
        solver = SimplexSolver(c, A, b)
    elif method == 'interior-point':
        solver = InteriorPointSolver(c, A, b)
    else:
        raise ValueError(f"Unknown method: {method}")

    return solver.solve()

def convert_to_standard_form(c: List[float], A: List[List[float]], b: List[float],
                             eq_constraints: List[bool] = None,
                             geq_constraints: List[bool] = None) -> Tuple[
    List[float], List[List[float]], List[float]]:
    """
    Convert a linear programming problem to standard form.

    Standard form: Maximize c^T x subject to Ax <= b and x >= 0

    Args:
        c: Coefficients of the objective function
        A: Constraint coefficients matrix
        b: Constraint right-hand side values
        eq_constraints: List indicating which constraints are equalities (=)
        geq_constraints: List indicating which constraints are inequalities (>=)

    Returns:
        Tuple of (c_new, A_new, b_new) in standard form
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    m, n = A.shape

    # Initialize constraint type flags if not provided
    if eq_constraints is None:
        eq_constraints = [False] * m
    if geq_constraints is None:
        geq_constraints = [False] * m

    # Convert equality constraints to two inequality constraints
    eq_indices = [i for i, is_eq in enumerate(eq_constraints) if is_eq]
    num_eq = len(eq_indices)

    if num_eq > 0:
        # For each equality constraint Ax = b, add two inequality constraints:
        # Ax <= b and -Ax <= -b
        A_expanded = np.vstack([A] + [-A[eq_indices, :]])
        b_expanded = np.hstack([b] + [-b[eq_indices]])

        # Update constraint type flags
        eq_constraints = [False] * (m + num_eq)
        geq_constraints = [geq for i, geq in enumerate(geq_constraints) if i not in eq_indices]
        geq_constraints += [False] * num_eq
    else:
        A_expanded = A
        b_expanded = b

    # Convert >= constraints to <= constraints
    geq_indices = [i for i, is_geq in enumerate(geq_constraints) if is_geq]

    if geq_indices:
        A_expanded[geq_indices, :] = -A_expanded[geq_indices, :]
        b_expanded[geq_indices] = -b_expanded[geq_indices]

    # Convert variables that can be negative to two non-negative variables
    # For each x_i, replace with x_i^+ - x_i^- where x_i^+, x_i^- >= 0
    c_new = np.hstack([c, -c])  # Coefficients for x^+ and x^-
    A_new = np.hstack([A_expanded, -A_expanded])

    return c_new.tolist(), A_new.tolist(), b_expanded.tolist()

def integer_linear_programming(c: List[float], A: List[List[float]], b: List[float],
                               method: str = 'branch-and-bound') -> Tuple[Optional[List[float]], Optional[float]]:
    """
    Solve an integer linear programming problem.

    Args:
        c: Coefficients of the objective function (maximize c^T x)
        A: Constraint coefficients matrix (Ax <= b)
        b: Constraint right-hand side values
        method: Solution method ('branch-and-bound')

    Returns:
        Tuple of (optimal_solution, optimal_value) if a solution exists,
        or (None, None) if the problem is infeasible or unbounded
    """
    if method != 'branch-and-bound':
        raise ValueError(f"Unknown method: {method}")

    # Solve the LP relaxation
    lp_solution, lp_value = solve_lp(c, A, b)

    if lp_solution is None:
        return None, None  # Problem is infeasible or unbounded

    # Check if all variables are integers
    if all(abs(x - round(x)) < 1e-6 for x in lp_solution):
        return [round(x) for x in lp_solution], lp_value

    # Find a non-integer variable
    for i, x in enumerate(lp_solution):
        if abs(x - round(x)) >= 1e-6:
            # Branch on this variable
            floor_x = math.floor(x)
            ceil_x = math.ceil(x)

            # Add constraint x_i <= floor(x_i)
            A_floor = A + [[0] * i + [1] + [0] * (len(c) - i - 1)]
            b_floor = b + [floor_x]

            # Recursively solve the branch
            sol_floor, val_floor = integer_linear_programming(c, A_floor, b_floor)

            # Add constraint x_i >= ceil(x_i) which is -x_i <= -ceil(x_i)
            A_ceil = A + [[0] * i + [-1] + [0] * (len(c) - i - 1)]
            b_ceil = b + [-ceil_x]

            # Recursively solve the branch
            sol_ceil, val_ceil = integer_linear_programming(c, A_ceil, b_ceil)

            # Return the better solution
            if sol_floor is None and sol_ceil is None:
                return None, None
            elif sol_floor is None:
                return sol_ceil, val_ceil
            elif sol_ceil is None:
                return sol_floor, val_floor
            else:
                if val_floor >= val_ceil:
                    return sol_floor, val_floor
                else:
                    return sol_ceil, val_ceil

    # Should not reach here
    return None, None