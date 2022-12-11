from unnet.nn import Neuron


def test_rand_neuron():
    neuron = Neuron.rand_neuron(2)
    assert len(neuron.weights) == 2
    for w in neuron.weights:
        assert -1.0 < w.value < 1.0


def test_neuron_output():
    neuron = Neuron(weights=[0.5, 2], bias=0.2)
    result = neuron.out([2.0, 3.0])
    assert result.value == 7.2
