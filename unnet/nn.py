from __future__ import annotations

import random
from dataclasses import dataclass

from .grad import Node


@dataclass
class Neuron:
    bias: Node
    weights: list[Node]

    def __init__(self, weights: list[float], bias: float):
        self.weights = [Node(w) for w in weights]
        self.bias = Node(bias)

    def out(self, inputs: list[float]):
        # TODO: Implement an activation functions
        return sum((w * x for w, x in zip(self.weights, inputs)), self.bias)

    @classmethod
    def rand_neuron(cls, num_inputs: int, bias: float = 0.0):
        weights = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        return cls(weights=weights, bias=bias)
