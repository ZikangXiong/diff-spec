from abc import ABC, abstractmethod
from typing import TypeVar

import torch as th


class PredicateBase(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def eval(self, xs: th.Tensor) -> th.Tensor:
        pass

    def __repr__(self) -> str:
        return self.name


AST = TypeVar("AST", list, PredicateBase)

BINARY_OPS = ("&", "|", "=>")
UNARY_OPS = ("!", "∃", "∀")


class FOL:
    def __init__(self, ast: AST, hardness: float = 10.0) -> None:
        self.ast = ast
        self.hardness = hardness

    def __and__(self, other: "FOL") -> "FOL":
        return FOL([BINARY_OPS[0], self.ast, other.ast], self.hardness)

    def __or__(self, other: "FOL") -> "FOL":
        return FOL([BINARY_OPS[1], self.ast, other.ast], self.hardness)

    def __rshift__(self, other: "FOL") -> "FOL":
        return FOL([BINARY_OPS[2], self.ast, other.ast], self.hardness)

    def __invert__(self) -> "FOL":
        return FOL([UNARY_OPS[0], self.ast], self.hardness)

    def exists(self) -> "FOL":
        return FOL([UNARY_OPS[1], self.ast], self.hardness)

    def forall(self) -> "FOL":
        return FOL([UNARY_OPS[2], self.ast], self.hardness)

    def eval(self, xs: th.Tensor) -> th.Tensor:
        return self._eval(self.ast, xs)

    def _eval(self, ast: AST, xs: th.Tensor) -> th.Tensor:
        if issubclass(type(ast), PredicateBase):
            return ast.eval(xs)
        elif isinstance(ast, list):
            if ast[0] in BINARY_OPS:
                return self._eval_binary(ast, xs)
            elif ast[0] in UNARY_OPS:
                return self._eval_unary(ast, xs)
            else:
                raise ValueError(f"Unknown operator {ast[0]}")
        else:
            raise ValueError(f"Unknown type {type(ast)}")

    def _eval_binary(self, ast: AST, xs: th.Tensor) -> th.Tensor:
        if ast[0] == "&":
            return self._eval_and(ast, xs)
        elif ast[0] == "|":
            return self._eval_or(ast, xs)
        elif ast[0] == "=>":
            return self._eval_implies(ast, xs)
        else:
            raise ValueError(f"Unknown operator {ast[0]}")

    def _eval_unary(self, ast: AST, xs: th.Tensor) -> th.Tensor:
        if ast[0] == "!":
            return self._eval_not(ast, xs)
        elif ast[0] == "∃":
            return self._eval_exists(ast, xs)
        elif ast[0] == "∀":
            return self._eval_forall(ast, xs)
        else:
            raise ValueError(f"Unknown operator {ast[0]}")

    def _eval_and(self, ast: AST, xs: th.Tensor) -> th.Tensor:
        left = self._eval(ast[1], xs)
        right = self._eval(ast[2], xs)

        return self._tensor_min(th.stack([left, right], dim=-1), dim=-1)

    def _eval_or(self, ast: AST, xs: th.Tensor) -> th.Tensor:
        left = self._eval(ast[1], xs)
        right = self._eval(ast[2], xs)

        return self._tensor_max(th.stack([left, right], dim=-1), dim=-1)

    def _eval_implies(self, ast: AST, xs: th.Tensor) -> th.Tensor:
        left = self._eval(ast[1], xs)
        right = self._eval(ast[2], xs)

        return self._tensor_max(th.stack([-left, right], dim=-1), dim=-1)

    def _eval_not(self, ast: AST, xs: th.Tensor) -> th.Tensor:
        return -self._eval(ast[1], xs)

    def _eval_exists(self, ast: AST, xs: th.Tensor) -> th.Tensor:
        return self._tensor_max(self._eval(ast[1], xs), dim=-1)

    def _eval_forall(self, ast: AST, xs: th.Tensor) -> th.Tensor:
        return self._tensor_min(self._eval(ast[1], xs), dim=-1)

    def _tensor_min(self, tensor: th.Tensor, dim=-1) -> th.Tensor:
        ratio = th.softmax(tensor * -self.hardness, dim=dim)
        return th.sum(tensor * ratio, dim=dim)

    def _tensor_max(self, tensor: th.Tensor, dim=-1) -> th.Tensor:
        ratio = th.softmax(tensor * self.hardness, dim=dim)
        return th.sum(tensor * ratio, dim=dim)

    def __repr__(self) -> str:
        stack = [self.ast]

        res = ""
        while len(stack) > 0:
            if issubclass(type(stack[-1]), PredicateBase):
                res += str(stack[-1])
                stack.pop()
            elif isinstance(stack[-1], str):
                if stack[-1] in BINARY_OPS:
                    res += " " + stack[-1] + " "
                elif stack[-1] in UNARY_OPS:
                    res += stack[-1] + " "
                else:
                    res += stack[-1]
                stack.pop()
            elif isinstance(stack[-1], list):
                if len(stack[-1]) == 3:
                    ele = stack.pop()
                    stack.append(")")
                    stack.append(ele[2])
                    stack.append(ele[0])
                    stack.append(ele[1])
                    stack.append("(")
                elif len(stack[-1]) == 2:
                    ele = stack.pop()
                    stack.append(")")
                    stack.append(ele[1])
                    stack.append(ele[0])
                    stack.append("(")
                else:
                    print(res)
                    raise SyntaxError("Invalid AST")

        return res

    def __call__(self, xs: th.Tensor) -> th.Tensor:
        return self.eval(xs)
