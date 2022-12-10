from __future__ import annotations


class Node:
    """Representation of an expression node capable of performing math operations and calculate the back propagation"""

    def __init__(self, value: int | float) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Node({self.value})"

    def __add__(self, other) -> Node:
        return Node(self.value + other.value)

    def __sub__(self, other) -> Node:
        return Node(self.value - other.value)

    def __mul__(self, other) -> Node:
        return Node(self.value * other.value)

    def __eq__(self, other) -> Node:
        return self.value == other.value

    # TODO: Track parents and calculate back propagation