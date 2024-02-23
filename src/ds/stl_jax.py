import io
import time
from abc import abstractmethod
from collections import deque
from contextlib import redirect_stdout
from typing import TypeVar, Tuple

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from stlpy.STL import LinearPredicate, NonlinearPredicate, STLTree
from torch import Tensor
from torch.nn.functional import softmax

from ds.utils import default_tensor

with redirect_stdout(io.StringIO()):
    from stlpy.solvers.base import STLSolver

import logging

from .stl import colored, HARDNESS, GurobiMICPSolver, STLSolver


class PredicateBase:
    def __init__(self, name: str):
        self.name = name

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
        res = torch.min(
            self.size_tensor / 2 - torch.abs(eval_path - self.cent_tensor), dim=-1
        )[0]

        return res

    def get_stlpy_form(self) -> STLTree:
        bounds = np.stack(
            [self.cent - self.size / 2, self.cent + self.size / 2]
        ).T.flatten()
        return inside_rectangle_formula(bounds, 0, 1, 2, self.name)


class RectAvoidPredicte(PredicateBase):
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


def inside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being inside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return inside_rectangle:   An ``STLFormula`` specifying being inside the
                                rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = 1
    right = LinearPredicate(a1, y1_min)
    left = LinearPredicate(-a1, -y1_max)

    a2 = np.zeros((1, d))
    a2[:, y2_index] = 1
    top = LinearPredicate(a2, y2_min)
    bottom = LinearPredicate(-a2, -y2_max)

    # Take the conjuction across all the sides
    inside_rectangle = right & left & top & bottom

    # set the names
    if name is not None:
        inside_rectangle.__str__ = lambda: name
        inside_rectangle.__repr__ = lambda: name

    return inside_rectangle


def outside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being outside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return outside_rectangle:   An ``STLFormula`` specifying being outside the
                                 rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = 1
    right = LinearPredicate(a1, y1_max)
    left = LinearPredicate(-a1, -y1_min)

    a2 = np.zeros((1, d))
    a2[:, y2_index] = 1
    top = LinearPredicate(a2, y2_max)
    bottom = LinearPredicate(-a2, -y2_min)

    # Take the disjuction across all the sides
    outside_rectangle = right | left | top | bottom

    # set the names
    if name is not None:
        outside_rectangle.__str__ = lambda: name
        outside_rectangle.__repr__ = lambda: name

    return outside_rectangle


AST = TypeVar("AST", list, PredicateBase)


class STL:
    """
    Class for representing STL formulas.
    """

    def __init__(self, ast: AST):
        self.ast = ast
        self.single_operators = ("~", "G", "F")
        self.binary_operators = ("&", "|", "->", "U")
        self.sequence_operators = ("G", "F", "U")
        self.stlpy_form = None
        self.expr_repr = None
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
