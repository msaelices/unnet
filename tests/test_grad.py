from unnet.grad import Node


def test_add():
    assert Node(4.5) + Node(0) == Node(4.5)
    assert Node(4.5) + Node(0.5) == Node(5.0)
    assert Node(4.5) + Node(-1) == Node(3.5)
    n1, n2 = Node(1), Node(2)
    assert (n1 + n2).parents == (n1, n2)


def test_multiply():
    assert Node(0.0) * Node(2.0) == Node(0.0)
    assert Node(1.0) * Node(2.0) == Node(2.0)
    assert Node(1.1) * Node(2.1) == Node(2.3100000000000005)
    n1, n2 = Node(1), Node(2)
    assert (n1 * n2).parents == (n1, n2)


def test_backward_propagation():
    a = Node(2.0)
    b = Node(3.0)
    c = a * b
    d = Node(1.5)
    e = d + c + a
    e.backward()

    assert e.grad == 1.0
    assert d.grad == 1.0
    assert c.grad == 1.0
    assert b.grad == 2.0
    assert a.grad == 4.0
