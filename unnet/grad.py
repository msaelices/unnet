from __future__ import annotations

from dataclasses import dataclass

from .utils import walk


ADD = '+'
MUL = '*'


def _calculate_gradients(op: str, node: Node, other: Node, result: Node) -> float:
    match op:
        case '+':
            node.grad += result.grad
            other.grad += result.grad
        case '*':
            node.grad += other.value * result.grad
            other.grad += node.value * result.grad
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
        return f'{self.name if self.name else "Node"}({self.value})'

    def __hash__(self) -> int:
        return id(self)

    def __add__(self, other) -> Node:
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value + other.value, name=f'{self.name} {ADD} {other.name}', op=ADD, parents=(self, other))

    def __mul__(self, other) -> Node:
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value * other.value, name=f'{self.name} {MUL} {other.name}', op=MUL, parents=(self, other))

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
