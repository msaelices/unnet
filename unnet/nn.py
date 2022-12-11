from __future__ import annotations

import random
from dataclasses import dataclass

from .grad import Node


@dataclass
class Neuron:
    bias: Node
    weights: list[Node]

    def __init__(self, weights: list[float], bias: float):
        self.weights = [Node(w, name=f'w{i}') for i, w in enumerate(weights, 1)]
        self.bias = Node(bias, name='bias')

    def __repr__(self) -> str:
        return f'Neuron(num_inputs={len(self.weights)})'

    def out(self, inputs: list[float]) -> Node:
        # TODO: Implement an activation functions
        input_nodes = [x if isinstance(x, Node) else Node(x, name=f'x{i}') for i, x in enumerate(inputs)]
        # Separate the first input and weight so the sum function give us a closer plot when drawing the resulting node
        w1, *remaining_weights = self.weights
        x1, *remaining_inputs = input_nodes
        return sum((w * x for w, x in zip(remaining_weights, remaining_inputs)), start=w1 * x1) + self.bias

    @classmethod
    def rand_neuron(cls, num_inputs: int, bias: float = 0.0):
        weights = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        return cls(weights=weights, bias=bias)


@dataclass
class Layer:
    neurons: list[Neuron]

    def __repr__(self):
        return f'Layer(neurons={", ".join(str(n) for n in self.neurons)})'

    def out(self, inputs: list[float]) -> list[Node]:
        return [n.out(inputs) for n in self.neurons]

    @classmethod
    def rand_layer(cls, num_neurons: int, num_inputs: int):
        neurons = [Neuron.rand_neuron(num_inputs) for _ in range(num_neurons)]
        return cls(neurons)


@dataclass
class Network:
    layers: list[Layer]

    def __repr__(self):
        return f'Network(num_layers={len(self.layers)})'

    def out(self, inputs: list[float]) -> list[Node]:
        for layer in self.layers:
            inputs = layer.out(inputs)
        return inputs
