from pytest import approx
from unnet.nn import Neuron, Layer, Network


def test_rand_neuron():
    neuron = Neuron.rand_neuron(2)
    assert len(neuron.weights) == 2
    for w in neuron.weights:
        assert -1.0 < w.value < 1.0


def test_neuron_output():
    neuron = Neuron(weights=[0.5, 2], bias=0.2)
    result = neuron([2.0, 3.0])
    assert result.value == 0.9999988852198828


def test_layer_output():
    neuron1 = Neuron(weights=[0.5, 0.8], bias=0.2)
    neuron2 = Neuron(weights=[0.2, -0.2], bias=0.9)
    layer = Layer([neuron1, neuron2])
    results = layer([2.0, 3.0])
    assert results[0].value == 0.9985079423323266
    assert results[1].value == 0.6043677771171636


def test_network_output():
    neuron1 = Neuron(weights=[0.5, 0.8], bias=0.2)
    neuron2 = Neuron(weights=[0.2, -0.2], bias=0.9)
    neuron3 = Neuron(weights=[-0.5, 0.3], bias=-0.2)
    layer1 = Layer([neuron1, neuron2])
    layer2 = Layer([neuron3])
    network = Network([layer1, layer2])
    result = network([2.0, 3.0])
    assert result.value == -0.4761113481082351


def test_network_create():
    network = Network.create(3, [2, 1])
    assert len(network.layers) == 2
    assert len(network.layers[0].neurons) == 2
    assert len(network.layers[1].neurons) == 1


def test_network_train():
    # training a neural network to choose the minimum of three numbers
    xs = [
        [3.0, 2.0, -1.0],
        [1.0, 0.0, -0.5],
        [0.5, 2.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0],
    ]  # inputs
    ys = [-1.0, -0.5, 0.5, 0.0, 1.0, 0.0]  # targets (minimum of each input)
    # neural network with 3 inputs, 3 hidden neurons, and 1 output neuron
    network = Network.create(3, [3, 1])

    loss_list = list(network.train_gen(training_data=xs, desired_output=ys, steps=1000))
    for loss in loss_list:
        print(loss)  # print loss for each step (hidden by pytest when test passes)

    assert loss_list[-1] < loss_list[0]
    assert loss_list[-1] < 0.1
    # check the output of the trained neural network with the first input
    assert network([3.0, 2.0, -1.0]).value == approx(-1.0, 0.1)
