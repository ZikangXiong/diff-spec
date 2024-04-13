import gurobipy as gp
import io
import numpy as np
import time
import torch
from abc import abstractmethod
from collections import deque
from contextlib import redirect_stdout
from gurobipy import GRB
from stlpy.STL import LinearPredicate, NonlinearPredicate, STLTree
from stlpy.systems import LinearSystem
from torch import Tensor
from torch.nn.functional import softmax
from typing import TypeVar, Tuple

from ds.utils import default_tensor, colored, HARDNESS, IMPLIES_TRICK, set_hardness, outside_rectangle_formula, \
    inside_rectangle_formula

with redirect_stdout(io.StringIO()):
    from stlpy.solvers.base import STLSolver

import logging


class GurobiMICPSolver(STLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`,
    solve the optimization problem

    .. math::

        \min & -\\rho^{\\varphi}(y_0,y_1,\dots,y_T) + \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = A x_t + B u_t

        & y_{t} = C x_t + D u_t

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    with Gurobi using mixed-integer convex programming. This gives a globally optimal
    solution, but may be computationally expensive for long and complex specifications.

    .. note::

        This class implements the algorithm described in

        Belta C, et al.
        *Formal methods for control synthesis: an optimization perspective*.
        Anual Review of Control, Robotics, and Autonomous Systems, 2019.

    :param spec:            An :class:`.STLFormula` describing the specification.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    :param presolve:        (optional) A boolean indicating whether to use Gurobi's
                            presolve routines. Default is ``True``.
    :param verbose:         (optional) A boolean indicating whether to print detailed
                            solver info. Default is ``True``.
    """

    def __init__(
            self,
            spec,
            sys,
            x0,
            T,
            M=1000,
            robustness_cost=True,
            presolve=True,
            verbose=True,
    ):
        assert M > 0, "M should be a (large) positive scalar"
        super().__init__(spec, sys, x0, T, verbose)

        self.M = float(M)
        self.presolve = presolve

        # Set up the optimization problem
        self.model = gp.Model("STL_MICP")

        # Store the cost function, which will added to self.model right before solving
        self.cost = 0.0

        # Set some model parameters
        if not self.presolve:
            self.model.setParam("Presolve", 0)
        if not self.verbose:
            self.model.setParam("OutputFlag", 0)

        if self.verbose:
            print("Setting up optimization problem...")
            st = time.time()  # for computing setup time

        # Create optimization variables
        self.y = self.model.addMVar((self.sys.p, self.T), lb=-float("inf"), name="y")
        self.x = self.model.addMVar((self.sys.n, self.T), lb=-float("inf"), name="x")
        self.u = self.model.addMVar((self.sys.m, self.T), lb=-float("inf"), name="u")
        self.rho = self.model.addMVar(
            1, name="rho", lb=0.0
        )  # lb sets minimum robustness

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints()
        self.AddSTLConstraints()
        self.AddRobustnessConstraint()
        if robustness_cost:
            self.AddRobustnessCost()

        if self.verbose:
            print(f"Setup complete in {time.time() - st} seconds.")

    def AddControlBounds(self, u_min, u_max):
        for t in range(self.T):
            self.model.addConstr(u_min <= self.u[:, t])
            self.model.addConstr(self.u[:, t] <= u_max)

    def AddStateBounds(self, x_min, x_max):
        for t in range(self.T):
            self.model.addConstr(x_min <= self.x[:, t])
            self.model.addConstr(self.x[:, t] <= x_max)

    def AddQuadraticCost(self, Q, R):
        self.cost += self.x[:, 0] @ Q @ self.x[:, 0] + self.u[:, 0] @ R @ self.u[:, 0]
        for t in range(1, self.T):
            self.cost += (
                    self.x[:, t] @ Q @ self.x[:, t] + self.u[:, t] @ R @ self.u[:, t]
            )

    def AddRobustnessCost(self):
        self.cost -= 1 * self.rho

    def AddRobustnessConstraint(self, rho_min=0.0):
        self.model.addConstr(self.rho >= rho_min)

    def Solve(self, time_limit=None, threads=0):
        # Set the cost function now, right before we solve.
        # This is needed since model.setObjective resets the cost.
        self.model.setObjective(self.cost, GRB.MINIMIZE)
        if time_limit is not None:
            self.model.setParam("TimeLimit", time_limit)
        if threads > 0:
            self.model.setParam("Threads", threads)

        # Do the actual solving
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            if self.verbose:
                print("\nOptimal Solution Found!\n")
            x = self.x.X
            u = self.u.X
            rho = self.rho.X[0]

            # Report optimal cost and robustness
            if self.verbose:
                print("Solve time: ", self.model.Runtime)
                print("Optimal robustness: ", rho)
                print("")
        else:
            if self.verbose:
                print(f"\nOptimization failed with status {self.model.status}.\n")
            x = None
            u = None
            rho = -np.inf

        return (x, u, rho, self.model.Runtime)

    def AddDynamicsConstraints(self):
        # Initial condition
        self.model.addConstr(self.x[:, 0] == self.x0)

        # Dynamics
        for t in range(self.T - 1):
            self.model.addConstr(
                self.x[:, t + 1]
                == self.sys.A @ self.x[:, t] + self.sys.B @ self.u[:, t]
            )

            self.model.addConstr(
                self.y[:, t] == self.sys.C @ self.x[:, t] + self.sys.D @ self.u[:, t]
            )

        self.model.addConstr(
            self.y[:, self.T - 1]
            == self.sys.C @ self.x[:, self.T - 1] + self.sys.D @ self.u[:, self.T - 1]
        )

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Recursively traverse the tree defined by the specification
        # to add binary variables and constraints that ensure that
        # rho is the robustness value
        z_spec = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
        self.AddSubformulaConstraints(self.spec, z_spec, 0)
        self.model.addConstr(z_spec == 1)

    def AddSubformulaConstraints(self, formula, z, t):
        """
        Given an STLFormula (formula) and a binary variable (z),
        add constraints to the optimization problem such that z
        takes value 1 only if the formula is satisfied (at time t).

        If the formula is a predicate, this constraint uses the "big-M"
        formulation

            A[x(t);u(t)] - b + (1-z)M >= 0,

        which enforces A[x;u] - b >= 0 if z=1, where (A,b) are the
        linear constraints associated with this predicate.

        If the formula is not a predicate, we recursively traverse the
        subformulas associated with this formula, adding new binary
        variables z_i for each subformula and constraining

            z <= z_i  for all i

        if the subformulas are combined with conjunction (i.e. all
        subformulas must hold), or otherwise constraining

            z <= sum(z_i)

        if the subformulas are combined with disjuction (at least one
        subformula must hold).
        """
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, LinearPredicate):
            # a.T*y - b + (1-z)*M >= rho
            self.model.addConstr(
                formula.a.T @ self.y[:, t] - formula.b + (1 - z) * self.M >= self.rho
            )

            # Force z to be binary
            b = self.model.addMVar(1, vtype=GRB.BINARY)
            self.model.addConstr(z == b)

        elif isinstance(formula, NonlinearPredicate):
            raise TypeError(
                "Mixed integer programming does not support nonlinear predicates"
            )

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
                    t_sub = formula.timesteps[i]  # the timestep at which this formula
                    # should hold
                    self.AddSubformulaConstraints(subformula, z_sub, t + t_sub)
                    self.model.addConstr(z <= z_sub)

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
                    z_subs.append(z_sub)
                    t_sub = formula.timesteps[i]
                    self.AddSubformulaConstraints(subformula, z_sub, t + t_sub)
                self.model.addConstr(z <= sum(z_subs))


class StlpySolver:
    """
    A class for solving STL specifications using mixed-integer programming.
    """

    def __init__(self, space_dim: int):
        self.space_dim = space_dim
        self._ctrl_sys = self._get_ctrl_system(space_dim)

    @staticmethod
    def _get_ctrl_system(dim: int):
        A = np.eye(dim)
        B = np.eye(dim)
        C = np.eye(dim)
        D = np.zeros([dim, dim])
        sys = LinearSystem(A, B, C, D)

        return sys

    def solve_stlpy_formula(
            self,
            spec: STLTree,
            x0: np.ndarray,
            total_time: int,
            solver_name="gurobi",
            u_bound: tuple = (-20.0, 20.0),
            rho_min: float = 0.1,
            energy_obj: bool = True,
            time_limit=20,
            threads=1,
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve the STL formula
        spec: stlpy formula
        x0: initial state
        total_time: total time steps
        solver_name: solver name
        u_bound: control input bound
        rho_min: robustness lower bound
        energy_obj: whether to minimize the energy
        time_limit: time limit for the solver
        threads: number of threads for the solver
        """
        if solver_name == "gurobi":
            solver = GurobiMICPSolver(
                spec, self._ctrl_sys, x0, total_time, verbose=False
            )
        else:
            raise NotImplementedError(f"{solver_name} is not supported yet")

        if energy_obj:
            solver.AddQuadraticCost(
                Q=np.diag(0.0 * np.random.random(self.space_dim)),
                R=np.eye(self.space_dim),
            )
        solver.AddControlBounds(*u_bound)
        solver.AddRobustnessConstraint(rho_min=rho_min)
        x, u, rho, solve_t = solver.Solve(time_limit=time_limit, threads=threads)

        info = dict(u=u, rho=rho, solve_t=solve_t)

        if x is not None:
            x = np.array(x).T
        return x, info


class PredicateBase:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)

    def eval_at_t(self, path: Tensor, t: int = 0) -> Tensor:
        return self.eval_whole_path(path, t, t + 1)[:, 0]

    @abstractmethod
    def eval_whole_path(
            self, path: Tensor, start_t: int = 0, end_t: int = None
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_stlpy_form(self) -> STLTree:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name


class RectReachPredicate(PredicateBase):
    """
    Rectangle reachability predicate
    """

    def __init__(self, cent: np.ndarray, size: np.ndarray, name: str, shrink_factor: float = 0.5):
        """
        :param cent: center of the rectangle
        :param size: bound of the rectangle
        :param name: name of the predicate
        """
        super().__init__(name)
        self.cent = cent
        self.size = size

        self.cent_tensor = default_tensor(cent)
        self.size_tensor = default_tensor(size)
        self.shrink_factor = shrink_factor  # shrink the rectangle to make it more conservative in STLpy
        self.logger.info(f"shrink factor: {shrink_factor}")

    def eval_whole_path(
            self, path: Tensor, start_t: int = 0, end_t: int = None
    ) -> Tensor:
        assert len(path.shape) == 3, "motion must be in batch"
        eval_path = path[:, start_t:end_t]
        res = torch.min(
            self.size_tensor / 2 - torch.abs(eval_path - self.cent_tensor), dim=-1
        )[0]

        return res

    def get_stlpy_form(self) -> STLTree:
        bounds = np.stack(
            [self.cent - self.size * self.shrink_factor / 2, self.cent + self.size * self.shrink_factor / 2]
        ).T.flatten()
        return inside_rectangle_formula(bounds, 0, 1, 2, self.name)


class RectAvoidPredicate(PredicateBase):
    """
    Rectangle avoidance predicate
    """

    def __init__(self, cent: np.ndarray, size: np.ndarray, name: str):
        """
        :param cent: center of the rectangle
        :param size: bound of the rectangle
        :param name: name of the predicate
        """
        super().__init__(name)
        self.cent = cent
        self.size = size

        self.cent_tensor = default_tensor(cent)
        self.size_tensor = default_tensor(size)

    def eval_whole_path(
            self, path: Tensor, start_t: int = 0, end_t: int = None
    ) -> Tensor:
        assert len(path.shape) == 3, "motion must be in batch"
        eval_path = path[:, start_t:end_t]
        res = torch.max(
            torch.abs(eval_path - self.cent_tensor) - self.size_tensor / 2, dim=-1
        )[0]

        return res

    def get_stlpy_form(self) -> STLTree:
        bounds = np.stack(
            [self.cent - self.size / 2, self.cent + self.size / 2]
        ).T.flatten()
        return outside_rectangle_formula(bounds, 0, 1, 2, self.name)


AST = TypeVar("AST", list, PredicateBase)


class STL:
    """
    Class for representing STL formulas.
    """
    end_t: int

    def __init__(self, ast: AST):
        self.ast = ast
        self.single_operators = ("~", "G", "F")
        self.binary_operators = ("&", "|", "->", "U")
        self.sequence_operators = ("G", "F", "U")
        self.stlpy_form = None
        self.expr_repr = None
        self.end_t = None  # Populated when evaluating
        self.logger = logging.getLogger(__name__)

    """
    Syntax Functions
    """

    def __and__(self, other: "STL") -> "STL":
        ast = ["&", self.ast, other.ast]
        return STL(ast)

    def __or__(self, other: "STL") -> "STL":
        ast = ["|", self.ast, other.ast]
        return STL(ast)

    def __invert__(self) -> "STL":
        ast = ["~", self.ast]
        return STL(ast)

    def implies(self, other: "STL") -> "STL":
        ast = ["->", self.ast, other.ast]
        return STL(ast)

    def eventually(self, start: int, end: int):
        ast = ["F", self.ast, start, end]
        return STL(ast)

    def always(self, start: int, end: int) -> "STL":
        ast = ["G", self.ast, start, end]
        return STL(ast)

    def until(self, other: "STL", start: int, end: int) -> "STL":
        ast = ["U", self.ast, other.ast, start, end]
        return STL(ast)

    def eval(self, path: Tensor, t: int = 0) -> Tensor:
        return self._eval(self.ast, path, t)

    def end_time(self) -> int:
        """Get the end time of the formula efficiently."""
        if self.end_t is None:
            # Evaluate the formula to get the end time
            # Get max of binary tree at self.ast
            self.end_t = self._get_end_time(self.ast)
        return self.end_t

    def _get_end_time(self, ast: AST) -> int:
        """Get max time of the formula. Runs in O(n) time where n is the number of nodes. Runs once then memoizes."""
        if self._is_leaf(ast):
            return 1
        if ast[0] in self.sequence_operators:
            # The last two elements are the start and end times
            return ast[-1]
        # Is binary operator
        return max(self._get_end_time(ast[1]), self._get_end_time(ast[2]))

    def _eval(
            self, ast: AST, path: Tensor, start_t: int = 0, end_t: int = None
    ) -> Tensor:
        if self._is_leaf(ast):
            return ast.eval_at_t(path, start_t)

        if ast[0] in self.sequence_operators:
            # NOTE: Overwrite start_t and end_t
            # this will allow access the elements after previous end_t
            start_t, end_t = start_t + ast[-2], start_t + ast[-1]
            if end_t > path.shape[1]:
                self.logger.warning("end_t is larger than motion length")

        if ast[0] == "&":
            res = self._eval_and(ast[1], ast[2], path, start_t, end_t)
        elif ast[0] == "|":
            res = self._eval_or(ast[1], ast[2], path, start_t, end_t)
        elif ast[0] == "~":
            res = self._eval_not(ast[1], path, start_t, end_t)
        elif ast[0] == "->":
            res = self._eval_implies(ast[1], ast[2], path, start_t, end_t)
        elif ast[0] == "G":
            res = self._eval_always(ast[1], path, start_t, end_t)
        elif ast[0] == "F":
            res = self._eval_eventually(ast[1], path, start_t, end_t)
        elif ast[0] == "U":
            res = self._eval_until(ast[1], ast[2], path, start_t, end_t)
        else:
            raise ValueError(f"Unknown operator {ast[0]}")

        return res

    def _eval_and(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: Tensor,
            start_t: int = 0,
            end_t: int = None,
    ) -> Tensor:
        return self._tensor_min(
            torch.stack(
                [
                    self._eval(sub_form1, path, start_t, end_t),
                    self._eval(sub_form2, path, start_t, end_t),
                ],
                dim=-1,
            ),
            dim=-1,
        )

    def _eval_or(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: Tensor,
            start_t: int = 0,
            end_t: int = None,
    ) -> Tensor:
        return self._tensor_max(
            torch.stack(
                [
                    self._eval(sub_form1, path, start_t, end_t),
                    self._eval(sub_form2, path, start_t, end_t),
                ],
                dim=-1,
            ),
            dim=-1,
        )

    def _eval_not(self, ast: AST, path: Tensor, start_t: int, end_t: int) -> Tensor:
        return -self._eval(ast, path, start_t, end_t)

    def _eval_implies(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: Tensor,
            start_t: int = 0,
            end_t: int = None,
    ) -> Tensor:
        if IMPLIES_TRICK:
            return self._eval(sub_form1, path, start_t, end_t) * self._eval(
                sub_form2, path, start_t, end_t
            )
        return self._eval_or(["~", sub_form1], sub_form2, path, start_t, end_t)

    def _eval_always(
            self, sub_form: AST, path: Tensor, start_t: int, end_t: int
    ) -> Tensor:
        if self._is_leaf(sub_form):
            return self._tensor_min(
                sub_form.eval_whole_path(path[:, start_t:end_t]), dim=-1
            )

        # unroll always
        val_per_time = torch.stack(
            [
                self._eval(sub_form, path, start_t=start_t + t, end_t=end_t)
                for t in range(end_t - start_t)
            ],
            dim=-1,
        )

        return self._tensor_min(val_per_time, dim=-1)

    def _eval_eventually(
            self, sub_form: AST, path: Tensor, start_t: int = 0, end_t: int = None
    ) -> Tensor:
        if self._is_leaf(sub_form):
            return self._tensor_max(
                sub_form.eval_whole_path(path[:, start_t:end_t]), dim=-1
            )

        # unroll eventually
        val_per_time = torch.stack(
            [
                self._eval(sub_form, path, start_t=start_t + t, end_t=end_t)
                for t in range(end_t - start_t)
            ],
            dim=-1,
        )

        return self._tensor_max(val_per_time, dim=-1)

    def _eval_until(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: Tensor,
            start_t: int = 0,
            end_t: int = None,
    ) -> Tensor:
        if self._is_leaf(sub_form2):
            till_pred = sub_form2.eval_whole_path(path[:, start_t:end_t])
        else:
            till_pred = torch.stack(
                [
                    self._eval(sub_form2, path, start_t=t, end_t=end_t)
                    for t in range(end_t - start_t)
                ],
                dim=-1,
            )
        # mask condition, once condition > 0 (after until True),
        # the right sequence is no longer considered
        cond = (till_pred > 0).int()
        index = torch.argmax(cond, dim=-1)
        for i in range(cond.shape[0]):
            cond[i, index[i]:] = 1.0
        cond = ~cond.bool()
        till_pred = torch.where(cond, till_pred, default_tensor(1))

        if self._is_leaf(sub_form1):
            res = sub_form1.eval_whole_path(path[:, start_t:end_t])
        else:
            res = torch.stack(
                [
                    self._eval(sub_form1, path, start_t=t, end_t=end_t)
                    for t in range(end_t - start_t)
                ],
                dim=-1,
            )
        res = torch.where(cond, res, default_tensor(-1))

        # when cond < 0, res should always > 0 to be hold
        return self._tensor_min(-res * till_pred, dim=-1)

    def get_stlpy_form(self):
        # catch already converted form
        if self.stlpy_form is None:
            self.stlpy_form = self._to_stlpy(self.ast)

        return self.stlpy_form

    def _to_stlpy(self, ast) -> STLTree:
        if self._is_leaf(ast):
            ast: AST = ast
            self.stlpy_form = ast.get_stlpy_form()
            return self.stlpy_form

        if ast[0] == "~":
            self.stlpy_form = self._convert_not(ast)
        elif ast[0] == "G":
            self.stlpy_form = self._convert_always(ast)
        elif ast[0] == "F":
            self.stlpy_form = self._convert_eventually(ast)
        elif ast[0] == "&":
            self.stlpy_form = self._convert_and(ast)
        elif ast[0] == "|":
            self.stlpy_form = self._convert_or(ast)
        elif ast[0] == "->":
            self.stlpy_form = self._convert_implies(ast)
        elif ast[0] == "U":
            self.stlpy_form = self._convert_until(ast)
        else:
            raise ValueError(f"Unknown operator {ast[0]}")

        return self.stlpy_form

    def _convert_not(self, ast):
        sub_form = self._to_stlpy(ast[1])
        return sub_form.negation()

    def _convert_and(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1 & sub_form_2

    def _convert_or(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1 | sub_form_2

    def _convert_implies(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1.negation() | sub_form_2

    def _convert_eventually(self, ast):
        sub_form = self._to_stlpy(ast[1])
        return sub_form.eventually(ast[2], ast[3])

    def _convert_always(self, ast):
        sub_form = self._to_stlpy(ast[1])
        return sub_form.always(ast[2], ast[3])

    def _convert_until(self, ast):
        sub_form_1 = self._to_stlpy(ast[1])
        sub_form_2 = self._to_stlpy(ast[2])
        return sub_form_1.until(sub_form_2, ast[3], ast[4])

    @staticmethod
    def _is_leaf(ast: AST):
        return issubclass(type(ast), PredicateBase)

    def _tensor_min(self, tensor: Tensor, dim=-1) -> Tensor:
        ratio = softmax(tensor * -HARDNESS, dim=dim)
        return torch.sum(tensor * ratio, dim=dim)

    def _tensor_max(self, tensor: Tensor, dim=-1) -> Tensor:
        ratio = softmax(tensor * HARDNESS, dim=dim)
        return torch.sum(tensor * ratio, dim=dim)

    def simplify(self):
        if self.stlpy_form is None:
            self.get_stlpy_form()
        self.stlpy_form.simplify()

    def __repr__(self):
        if self.expr_repr is not None:
            return self.expr_repr

        single_operators = ("~", "G", "F")
        binary_operators = ("&", "|", "->", "U")
        time_bounded_operators = ("G", "F", "U")

        # traverse ast
        operator_stack = [self.ast]
        expr = ""
        cur = self.ast

        def push_stack(ast):
            if isinstance(ast, str) and ast in time_bounded_operators:
                time_window = f"[{cur[-2]}, {cur[-1]}]"
                operator_stack.append(time_window)
            operator_stack.append(ast)

        while operator_stack:
            cur = operator_stack.pop()
            if self._is_leaf(cur):
                expr += cur.__str__()
            elif isinstance(cur, str):
                if cur == "(" or cur == ")":
                    expr += cur
                elif cur.startswith("["):
                    expr += colored(cur, "yellow") + " "
                else:
                    if cur in ("G", "F"):
                        if cur == "F":
                            expr += colored("F", "magenta")
                        else:
                            expr += colored(cur, "magenta")
                    elif cur in ("&", "|", "->", "U"):
                        expr += " " + colored(cur, "magenta")
                        if cur != "U":
                            expr += " "
                    elif cur in ("~",):
                        expr += colored(cur, "magenta")
            elif cur[0] in single_operators:
                # single operator
                if not self._is_leaf(cur[1]):
                    push_stack(")")
                push_stack(cur[1])
                if not self._is_leaf(cur[1]):
                    push_stack("(")
                push_stack(cur[0])
            elif cur[0] in binary_operators:
                # binary operator
                if not self._is_leaf(cur[2]) and cur[2][0] in binary_operators:
                    push_stack(")")
                    push_stack(cur[2])
                    push_stack("(")
                else:
                    push_stack(cur[2])
                push_stack(cur[0])
                if not self._is_leaf(cur[1]) and cur[1][0] in binary_operators:
                    push_stack(")")
                    push_stack(cur[1])
                    push_stack("(")
                else:
                    push_stack(cur[1])

        self.expr_repr = expr
        return expr

    def get_all_predicates(self):
        all_preds = []
        queue = deque([self.ast])

        while queue:
            cur = queue.popleft()

            if self._is_leaf(cur):
                all_preds.append(cur)
            elif cur[0] in self.single_operators:
                queue.append(cur[1])
            elif cur[0] in self.binary_operators:
                queue.append(cur[1])
                queue.append(cur[2])
            else:
                raise RuntimeError("Should never visit here")

        return all_preds
