from unnet.data import Node


def test_add():
    assert Node(4.5) + Node(0) == Node(4.5)
    assert Node(4.5) + Node(0.5) == Node(5.0)
    assert Node(4.5) + Node(-1) == Node(3.5)


def test_substraction():
    assert Node(4.5) - Node(0) == Node(4.5)
    assert Node(4.5) - Node(0.5) == Node(4.0)
    assert Node(4.5) - Node(-1) == Node(5.5)


def test_multiply():
    assert Node(0.0) * Node(2.0) == Node(0.0)
    assert Node(1.0) * Node(2.0) == Node(2.0)
    assert Node(1.1) * Node(2.1) == Node(2.3100000000000005)
