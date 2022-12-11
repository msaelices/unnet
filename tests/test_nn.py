from unnet.nn import Neuron, Layer, Network


def test_rand_neuron():
    neuron = Neuron.rand_neuron(2)
    assert len(neuron.weights) == 2
    for w in neuron.weights:
        assert -1.0 < w.value < 1.0


def test_neuron_output():
    neuron = Neuron(weights=[0.5, 2], bias=0.2)
    result = neuron.out([2.0, 3.0])
    assert result.value == 7.2


def test_layer_output():
    neuron1 = Neuron(weights=[0.5, 0.8], bias=0.2)
    neuron2 = Neuron(weights=[0.2, -0.2], bias=0.9)
    layer = Layer([neuron1, neuron2])
    results = layer.out([2.0, 3.0])
    assert results[0].value == 3.6000000000000005
    assert results[1].value == 0.7


def test_network_output():
    neuron1 = Neuron(weights=[0.5, 0.8], bias=0.2)
    neuron2 = Neuron(weights=[0.2, -0.2], bias=0.9)
    neuron3 = Neuron(weights=[-0.5, 0.3], bias=-0.2)
    neuron4 = Neuron(weights=[-0.2, 0.2], bias=0.4)
    layer1 = Layer([neuron1, neuron2])
    layer2 = Layer([neuron3, neuron4])
    network = Network([layer1, layer2])
    results = network.out([2.0, 3.0])
    assert results[0].value == -1.7900000000000003
    assert results[1].value == -0.18000000000000016
