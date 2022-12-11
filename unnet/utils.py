from __future__ import annotations

from graphviz import Digraph  # type: ignore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unnet.data import Node


def walk(n: Node) -> tuple[list, list]:
    nodes, edges = list(), list()

    if n not in nodes:
        nodes.append(n)
        for parent in n.parents:
            edges.append((parent, n))
            ns, es = walk(parent)
            nodes += ns
            edges += es

    return nodes, edges


def draw(graph: Node) -> Digraph:
    plot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = walk(graph)
    for n in nodes:
        n_name = str(hash(n))
        plot.node(
            name=n_name,
            label=f'{n.value} | g: {n.grad}',
            shape='record',
        )
        if n.op:
            plot.node(name=n_name + n.op, label=n.op, shape='circle')
            plot.edge(n_name + n.op, n_name)

    for n1, n2 in edges:
        plot.edge(str(hash(n1)), str(hash(n2)) + n2.op)

    return plot
