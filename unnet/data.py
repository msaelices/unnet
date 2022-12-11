from __future__ import annotations

from .utils import walk


ADD = '+'
SUB = '-'
MUL = '*'


def _calculate_gradients(op: str, node: Node, other: Node, result: Node) -> float:
    match op:
        case '+':
            node.grad = result.grad
            other.grad = result.grad
        case '*':
            node.grad = other.value * result.grad
            other.grad = node.value * result.grad
        case _:
            raise RuntimeError('Invalid operation')


class Node:
    """Representation of an expression node capable of performing math operations and calculate the back propagation"""

    def __init__(
        self,
        value: int | float,
        op: str | None = None,
        parents: tuple | None = None,
    ) -> None:
        self.value = value
        self.parents = parents or ()
        self.op = op
        self.grad: float = 0.0

    def __repr__(self) -> str:
        return f"Node({self.value})"
    
    def __hash__(self) -> int:
        return id(self)

    def __add__(self, other) -> Node:
        return Node(self.value + other.value, op=ADD, parents=(self, other))

    def __sub__(self, other) -> Node:
        return Node(self.value - other.value, op=SUB, parents=(self, other))

    def __mul__(self, other) -> Node:
        return Node(self.value * other.value, op=MUL, parents=(self, other))

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def backward(self):
        # the grad of myself is always 1
        self.grad = 1.0

        nodes, _ = walk(self)
        # generate the gradient of all my parents until we reach the end of the graph
        for n in nodes:
            if n.parents:
                _calculate_gradients(n.op, n.parents[0], n.parents[1], n)
