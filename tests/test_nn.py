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
    layer1 = Layer([neuron1, neuron2])
    layer2 = Layer([neuron3])
    network = Network([layer1, layer2])
    result = network.out([2.0, 3.0])
    assert result.value == -1.7900000000000003


def test_network_create():
    network = Network.create(3, [2, 1])
    assert len(network.layers) == 2
    assert len(network.layers[0].neurons) == 2
    assert len(network.layers[1].neurons) == 1


def test_network_train():
    neuron1 = Neuron(weights=[0.5, 0.8], bias=0.2)
    neuron2 = Neuron(weights=[0.2, -0.2], bias=0.9)
    neuron3 = Neuron(weights=[-0.5, 0.3], bias=-0.2)
    layer1 = Layer([neuron1, neuron2])
    layer2 = Layer([neuron3])
    network = Network([layer1, layer2])
    prev_loss = 1000000
    for loss in network.train_gen(training_data=[[1.2, 1.5], [3.4, 4.5]], desired_output=[1.0, 2.0]):
        assert loss < prev_loss
        prev_loss = loss
