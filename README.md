# unnet

`unnet` stands for `Micro Neural Network`

It is a proof-of-concept for self-learning of neural networks, so it simplifies and not try to be as efficient as possible. For example, it's not using tensors but simple floating point numbers.

**WARNING**: Please do not use in production code.

## Getting Started

### Forward and back propagations

Let's take this example:

```python
>>> from unnet.grad import Node
>>> a = Node(2.0, name='a')
>>> b = Node(3.0, name='b')
>>> c = Node(1.5, name='c')
>>> # forward propagation
>>> d = a * b
>>> e = d + c + a
>>> print(e)
a * b + c + a = 9.5
>>> # back propagation to calculate the gradients
>>> e.backward()
>>> print(e.grad)
1.0
>>> print(d.grad)
1.0
>>> print(c.grad)
1.0
>>> print(b.grad)
2.0
>>> print(a.grad)
4.0
```

We can imagine that these kind of operations could apply to a neural network performing a forward and backward propagations.

You can see a more detailed example in the [Grad Jupyter notebook](./notebooks/grad.ipynb), which includes plots of the calculated expressions

### Neural Networks

#### Working with individual neurons

Let's create an artificial neuron with 2 weights and calculate the output given 2 inputs:

```python
>>> from unnet.nn import Neuron
>>> neuron1 = Neuron(weights=[0.7, 0.8], bias=0.5)
>>> result = neuron1([2.0, 3.0])
>>> print(result)
w1 * x0 + w2 * x1 + bias = 4.300000000000001
```

#### Working with neural networks

Let's create an artificial neuron with 2 weights and calculate the output given 2 inputs:

```python
>>> neuron1 = Neuron(weights=[0.5, 0.8], bias=0.2)
>>> neuron2 = Neuron(weights=[0.2, -0.2], bias=0.9)
>>> neuron3 = Neuron(weights=[-0.5, 0.3], bias=-0.2)
>>> neuron4 = Neuron(weights=[-0.2, 0.2], bias=0.4)
>>> layer1 = Layer([neuron1, neuron2])
>>> layer2 = Layer([neuron3, neuron4])
>>> network = Network([layer1, layer2])
>>> results = network([2.0, 3.0])
>>> print(results[0].value, results[1].value)
-1.7900000000000003 -0.18000000000000016
```

You can see a more detailed example in the [NN Jupyter notebook](./notebooks/nn.ipynb), which includes plots of the calculated neuron

## Installation

### Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. If you don't have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/unnet.git
cd unnet
```

2. Install all dependencies (including dev dependencies):
```bash
uv sync
```


## Interactive Examples

- [Grad notebook](./notebooks/grad.ipynb) - Forward and backward propagation with visualizations
- [NN notebook](./notebooks/nn.ipynb) - Neural network training examples

For interactive exploration and testing:

```bash
uv run jupyter notebook notebooks/grad.ipynb
uv run jupyter notebook notebooks/nn.ipynb
```


## Contributing

We welcome contributions! If you'd like to contribute to unnet, please see our [Contributing Guide](CONTRIBUTING.md)

For questions or issues, please open an issue on GitHub.
