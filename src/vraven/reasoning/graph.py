"""Execution-summary graphs for VRAVEN traces."""

from __future__ import annotations

import networkx as nx

from vraven.core.trace import NeuralTrace


def build_reasoning_graph(trace: NeuralTrace) -> nx.DiGraph:
    """Build a sequential execution graph from captured modules.

    Edges represent observed execution order, not neuron-level connectivity or
    causality. This distinction is stored in graph metadata.
    """

    graph = nx.DiGraph(
        trace_id=trace.trace_id,
        semantics="captured_execution_summary",
        causal=False,
    )
    for layer in trace.layers:
        graph.add_node(
            layer.name,
            layer_type=layer.layer_type,
            execution_index=layer.execution_index,
            activation_energy=layer.activation_energy,
            signed_influence=layer.activation_gradient_signed,
            absolute_influence=layer.activation_gradient_influence,
            sparsity=layer.sparsity,
        )
    for previous, current in zip(trace.layers, trace.layers[1:], strict=False):
        graph.add_edge(
            previous.name,
            current.name,
            edge_type="execution_transition",
            weight=float(current.activation_gradient_influence or current.mean_energy or 0.0),
        )
    return graph


def graph_payload(trace: NeuralTrace) -> dict[str, object]:
    graph = build_reasoning_graph(trace)
    return nx.node_link_data(graph, edges="edges")
