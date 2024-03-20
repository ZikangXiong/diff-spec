import importlib
import io
import os
from abc import abstractmethod
from collections import deque
from contextlib import redirect_stdout
from typing import TypeVar

import jax
import numpy as np
import importlib
from jax.nn import softmax
from stlpy.STL import LinearPredicate, STLTree

os.environ["DIFF_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
import ds.utils as ds_utils
importlib.reload(ds_utils)  # Reload the module to change the backend

with redirect_stdout(io.StringIO()):
    pass

import logging

from .stl import colored, HARDNESS, IMPLIES_TRICK, set_hardness

# Replace with JAX
import jax.numpy as jnp


class PredicateBase:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)

    def eval_at_t(self, path: jnp.ndarray, t: int = 0) -> jnp.ndarray:
        return self.eval_whole_path(path, t, t + 1)[:, 0]

    @abstractmethod
    def eval_whole_path(
            self, path: jnp.ndarray, start_t: int = 0, end_t: int = None
    ) -> jnp.ndarray:
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

        self.cent_tensor = ds_utils.default_tensor(cent)
        self.size_tensor = ds_utils.default_tensor(size)
        self.shrink_factor = shrink_factor  # shrink the rectangle to make it more conservative
        self.logger.info(f"shrink factor: {shrink_factor}")

    def eval_whole_path(
            self, path: jnp.array, start_t: int = 0, end_t: int = None
    ) -> jnp.array:
        assert len(path.shape) == 3, "motion must be in batch"
        eval_path = path[:, start_t:end_t]
        res = jnp.min(
            self.size_tensor / 2 - jnp.abs(eval_path - self.cent_tensor), axis=-1
        )

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

        self.cent_tensor = ds_utils.default_tensor(cent)
        self.size_tensor = ds_utils.default_tensor(size)

    def eval_whole_path(
            self, path: jnp.array, start_t: int = 0, end_t: int = None
    ) -> jnp.array:
        assert len(path.shape) == 3, "motion must be in batch"
        eval_path = path[:, start_t:end_t]
        res = jnp.max(
            jnp.abs(eval_path - self.cent_tensor) - self.size_tensor / 2, axis=-1
        )

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

    def eval(self, path: jnp.array, t: int = 0) -> jnp.array:
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
            self, ast: AST, path: jnp.array, start_t: int = 0, end_t: int = None
    ) -> jnp.array:
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
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
    ) -> jnp.array:
        return self._tensor_min(
            jnp.stack(
                [
                    self._eval(sub_form1, path, start_t, end_t),
                    self._eval(sub_form2, path, start_t, end_t),
                ],
                axis=-1,
            ),
            axis=-1,
        )

    def _eval_or(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
    ) -> jnp.array:
        return self._tensor_max(
            jnp.stack(
                [
                    self._eval(sub_form1, path, start_t, end_t),
                    self._eval(sub_form2, path, start_t, end_t),
                ],
                axis=-1,
            ),
            axis=-1,
        )

    def _eval_not(self, ast: AST, path: jnp.array, start_t: int, end_t: int) -> jnp.array:
        return -self._eval(ast, path, start_t, end_t)

    def _eval_implies(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
    ) -> jnp.array:
        if IMPLIES_TRICK:
            return self._eval(sub_form1, path, start_t, end_t) * self._eval(
                sub_form2, path, start_t, end_t
            )
        return self._eval_or(["~", sub_form1], sub_form2, path, start_t, end_t)

    def _eval_always(
            self, sub_form: AST, path: jnp.array, start_t: int, end_t: int
    ) -> jnp.array:
        if self._is_leaf(sub_form):
            return self._tensor_min(
                sub_form.eval_whole_path(path[:, start_t:end_t]), axis=-1
            )

        # unroll always
        val_per_time = jnp.stack(
            [
                self._eval(sub_form, path, start_t=start_t + t, end_t=end_t)
                for t in range(end_t - start_t)
            ],
            axis=-1,
        )

        return self._tensor_min(val_per_time, axis=-1)

    def _eval_eventually(
            self, sub_form: AST, path: jnp.array, start_t: int = 0, end_t: int = None
    ) -> jnp.array:
        if self._is_leaf(sub_form):
            return self._tensor_max(
                sub_form.eval_whole_path(path[:, start_t:end_t]), axis=-1
            )

        # unroll eventually
        val_per_time = jnp.stack(
            [
                self._eval(sub_form, path, start_t=start_t + t, end_t=end_t)
                for t in range(end_t - start_t)
            ],
            axis=-1,
        )

        return self._tensor_max(val_per_time, axis=-1)

    def _eval_until(
            self,
            sub_form1: AST,
            sub_form2: AST,
            path: jnp.array,
            start_t: int = 0,
            end_t: int = None,
    ) -> jnp.array:
        if self._is_leaf(sub_form2):
            till_pred = sub_form2.eval_whole_path(path[:, start_t:end_t])
        else:
            till_pred = jnp.stack(
                [
                    self._eval(sub_form2, path, start_t=t, end_t=end_t)
                    for t in range(end_t - start_t)
                ],
                axis=-1,
            )
        # mask condition, once condition > 0 (after until True),
        # the right sequence is no longer considered
        cond = (till_pred > 0).int()
        index = jnp.argmax(cond, axis=-1)
        for i in range(cond.shape[0]):
            cond[i, index[i]:] = 1.0
        cond = ~cond.bool()
        till_pred = jnp.where(cond, till_pred, ds_utils.default_tensor(1))

        if self._is_leaf(sub_form1):
            res = sub_form1.eval_whole_path(path[:, start_t:end_t])
        else:
            res = jnp.stack(
                [
                    self._eval(sub_form1, path, start_t=t, end_t=end_t)
                    for t in range(end_t - start_t)
                ],
                axis=-1,
            )
        res = jnp.where(cond, res, ds_utils.default_tensor(-1))

        # when cond < 0, res should always > 0 to be hold
        return self._tensor_min(-res * till_pred, axis=-1)

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

    def _tensor_min(self, tensor: jnp.array, axis=-1) -> jnp.array:
        ratio = softmax(tensor * -HARDNESS, axis=axis)
        return jnp.sum(tensor * ratio, axis=axis)

    def _tensor_max(self, tensor: jnp.array, axis=-1) -> jnp.array:
        ratio = softmax(tensor * HARDNESS, axis=axis)
        return jnp.sum(tensor * ratio, axis=axis)

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
