from __future__ import annotations

import math
from dataclasses import dataclass

from .utils import walk


ADD = '+'
SUB = '-'
MUL = '*'
POW = '^'
TANH = 'tanh'


def _calculate_gradients(op: str, result: Node, node: Node, other: Node | None = None) -> None:
    match op:
        case '+':
            node.grad += result.grad
            other.grad += result.grad  # type: ignore
        case '-':
            node.grad -= result.grad
            other.grad -= result.grad  # type: ignore
        case '*':
            node.grad += other.value * result.grad  # type: ignore
            other.grad += node.value * result.grad  # type: ignore
        case '^':
            node.grad += other.value * node.value ** (other.value - 1) * result.grad  # type: ignore
        case 'tanh':
            node.grad += (1 - result.value**2) * result.grad
        case _:
            raise RuntimeError('Invalid operation')


@dataclass
class Node:
    """Representation of an expression node capable of performing math operations and calculate the back propagation"""

    value: int | float
    name: str = 'N/A'
    op: str | None = None
    parents: tuple = tuple()
    grad: int | float = 0.0

    def __repr__(self) -> str:
        return f'{self.name} = {self.value}'

    def __hash__(self) -> int:
        return id(self)

    def __add__(self, other) -> Node:
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value + other.value, name=f'{self.name} {ADD} {other.name}', op=ADD, parents=(self, other))

    def __radd__(self, other) -> Node:
        return self + other

    def __sub__(self, other) -> Node:
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value - other.value, name=f'{self.name} {SUB} {other.name}', op=ADD, parents=(self, other))

    def __mul__(self, other) -> Node:
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value * other.value, name=f'{self.name} {MUL} {other.name}', op=MUL, parents=(self, other))

    def __pow__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value**other.value, name=f'{self.name} {POW} {other.name}', op=POW, parents=(self, other))

    def __rpow__(self, other):
        return self**other

    def tanh(self):
        return Node(math.tanh(self.value), name=f'tanh({self.name})', op=TANH, parents=(self,))

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def backward(self):
        # the grad of myself is always 1
        self.grad = 1.0

        nodes, _ = walk(self)
        # generate the gradient of all my parents until we reach the end of the graph
        for n in nodes:
            if parents := n.parents:
                _calculate_gradients(n.op, n, *parents)
