from __future__ import annotations

from enum import Enum

ADD = '+'
SUB = '-'
MUL = '*'


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

    def __eq__(self, other) -> Node:
        return self.value == other.value

    # TODO: Calculate back propagation
