from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generator, cast

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

    @property
    def nodes(self):
        return self.weights + [self.bias]

    def out(self, inputs: list[float] | list[Node]) -> Node:
        input_nodes = cast(
            list[Node], [Node(x, name=f'x{i}') if isinstance(x, float) else x for i, x in enumerate(inputs)]
        )
        # Separate the first input and weight so the sum function give us a closer plot when drawing the resulting node
        w1, *remaining_weights = self.weights
        x1, *remaining_inputs = input_nodes
        result = sum((w * x for w, x in zip(remaining_weights, remaining_inputs)), start=w1 * x1) + self.bias

        # Apply the activation function to the result
        return result.tanh()

    @classmethod
    def rand_neuron(cls, num_inputs: int, bias: float = 0.0):
        weights = [random.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        return cls(weights=weights, bias=bias)


@dataclass
class Layer:
    neurons: list[Neuron]

    def __repr__(self):
        return f'Layer(neurons={", ".join(str(n) for n in self.neurons)})'

    @property
    def nodes(self):
        return [n for neuron in self.neurons for n in neuron.nodes]

    def out(self, inputs: list[float] | list[Node]) -> list[Node]:
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

    @property
    def nodes(self):
        return [n for layer in self.layers for n in layer.nodes]

    def out(self, inputs: list[float] | list[Node]) -> Node:
        for layer in self.layers:
            inputs = layer.out(inputs)
        # Note: this is only valid if the network has only one output
        return inputs[0]  # type: ignore

    def train_gen(self, training_data: list[list[float]], desired_output: list[float], steps: int = 20) -> Generator:
        for _ in range(steps):
            # forward step
            prediction = [self.out(x) for x in training_data]
            loss = sum((pred - out) ** 2 for out, pred in zip(desired_output, prediction))

            # back propagation
            for n in self.nodes:
                n.grad = 0.0
            loss.backward()

            # refine weights
            for n in self.nodes:
                n.value += -0.1 * n.grad

            yield loss.value

    @classmethod
    def create(cls, num_inputs: int, neurons: list[int]):
        """Create a neural network with num_inputs inputs, an array of intermediate neurons, and one output neuron"""
        assert neurons[-1] == 1  # only one output neuron

        layers = []
        layer_inputs = num_inputs
        for num_outputs in neurons:
            layers.append(Layer.rand_layer(num_inputs=layer_inputs, num_neurons=num_outputs))
            layer_inputs = num_outputs  # the inputs of the next layer are the outputs of the previous layer

        return cls(layers=layers)
