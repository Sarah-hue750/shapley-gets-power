#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as _np
import networkx as _nx
import itertools as it

from utils import distribution_factors

from utils.data_handling import (
    construct_incidencematrix_from_orientation,
    construct_incidencematrix_from_orientation_multigraph,
)


def generate_graph_from_similarity_matrix(
    edge_node_names: list,
    similarity_matrix: _np.ndarray,
    threshold: float | None = None,
) -> _nx.Graph:
    """Generate a graph that has the lines as nodes and
    the collectivity measures as edges."""

    if threshold is None:
        adjacency_matrix = similarity_matrix
    else:
        adjacency_matrix = _np.zeros(similarity_matrix.shape)
        adjacency_matrix[abs(similarity_matrix) > threshold] = similarity_matrix[
            abs(similarity_matrix) > threshold
        ]

    similarity_graph = _nx.from_numpy_array(adjacency_matrix)
    similarity_graph = _nx.relabel_nodes(
        similarity_graph, {i: name for i, name in enumerate(edge_node_names)}, copy=True
    )

    return similarity_graph


def alternative_collectivity_measure(
    graph: _nx.Graph,
):
    """Measure of collectivity that only includes part of the PTDF matrix."""

    BB = _nx.laplacian_matrix(graph, weight="weight").todense()

    if graph.is_multigraph():
        II = construct_incidencematrix_from_orientation_multigraph(
            graph, return_np_array=True
        )
    else:
        II = construct_incidencematrix_from_orientation(graph, return_np_array=True)

    try:
        # Implicit matrix inversion
        BB_inv_n_II = _np.linalg.solve(BB, II)
        collectivity_matrix = _np.linalg.multi_dot((II.T, BB_inv_n_II))

    except _np.linalg.LinAlgError:
        BB_inv = _np.linalg.pinv(BB)
        collectivity_matrix = _np.linalg.multi_dot((II.T, BB_inv, II))

    return collectivity_matrix


def get_edgenames_and_similarity_matrix(
    graph: _nx.Graph, atol_zero: float = 1e-20, similarity_measure: str = "predictor"
) -> _nx.Graph:
    """
    Cluster the LODFs based on their mutual relationships.
    This function will group LODFs that are similar or related to each other.
    """

    is_multigraph = graph.is_multigraph()

    # Remove dead ends
    graph_modified = graph.copy()
    one_degree_nodes = [node for node, degree in graph.degree() if degree == 1]
    while len(one_degree_nodes) > 0:
        graph_modified.remove_nodes_from(one_degree_nodes)
        one_degree_nodes = [
            node for node, degree in graph_modified.degree() if degree == 1
        ]

    assert _nx.is_connected(
        graph_modified
    ), "The graph must be connected after removing dead ends."

    # Bridges
    bridges = list(_nx.bridges(graph_modified))
    assert (
        len(bridges) == 0
    ), "The graph should not have bridges after removing dead ends."

    # Calculate LODFs
    lodf_matrix = distribution_factors.calculate_LODF_matrix(
        graph_modified, is_multigraph=is_multigraph
    )

    if similarity_measure == "predictor":
        similarity_matrix = _np.multiply(lodf_matrix, lodf_matrix.T)

    elif similarity_measure == "alternative":
        similarity_matrix = abs(alternative_collectivity_measure(graph_modified))

    else:
        raise ValueError(f"Unknown similarity measure: {similarity_measure}")

    _np.fill_diagonal(similarity_matrix, 0)

    ## Set small values to zero
    idx_small_values = _np.where(_np.abs(similarity_matrix) < atol_zero)
    similarity_matrix[idx_small_values] = 0.0

    if similarity_measure == "predictor":
        similarity_matrix = _np.sqrt(similarity_matrix)

    assert _np.all(
        similarity_matrix >= 0
    ), "The similarity matrix must be non-negative."

    if is_multigraph:
        edges_names = list(graph_modified.edges(keys=True))
    else:
        edges_names = list(graph_modified.edges())

    return edges_names, similarity_matrix


def extract_failed_edge_subgraph(
    graph: _nx.Graph, failed_edges: list, similarity_measure: str = "predictor"
) -> _nx.Graph:
    """Translate the graph to a similarity graph and extract a sugraph
    that only contains the edges that failed.

    Args:
        graph (_nx.Graph): Graph of the original network.
        failed_edges (list): List with edges that failed.
            Be sure to include keys if the graph is a multigraph.
        similarity_measure (str, optional): _description_. Defaults to 'predictor'.

    Returns:
        _nx.Graph: Graph only consisting of the nodes that correspond to the edges that failed
            in the original graph. Edges weights are the collectivity measures.
    """

    is_multigraph = graph.is_multigraph()

    edges_kwargs = {}
    if is_multigraph:
        edges_kwargs["keys"] = True

    assert all(
        edge in graph.edges(**edges_kwargs) for edge in failed_edges
    ), "All failed edges must be in the graph."

    assert len(set(failed_edges)) == len(
        failed_edges
    ), "All failed edges must be unique."

    # Get the collectivity measures
    edge_names, similarity_matrix = get_edgenames_and_similarity_matrix(
        graph, similarity_measure=similarity_measure
    )

    edge_failed_idx = [edge_names.index(edge) for edge in failed_edges]

    cut_similarity_matrix = similarity_matrix[edge_failed_idx, :][:, edge_failed_idx]

    cut_similarity_graph = generate_graph_from_similarity_matrix(
        failed_edges, cut_similarity_matrix
    )

    return cut_similarity_graph


def knn_failure_subgraphs(
    graph: _nx.Graph,
    edge_failed: list,
    similarity_measure: str = "predictor",
    nearest_neighbors: int = 8,
):
    """Get for each failed edge the k-nearest neighbors
    based on the collectivity measure.
    Args:
        graph (_nx.Graph): The input graph to be clustered.
        edge_failed (list): The list of failed edges to be considered as nodes.
        similarity_measure (str, optional): The similarity measure to be used. Defaults to "predictor".
        nearest_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 8.
    Returns:
        knn_dict (dict): A dictionary where keys are failed edges and values are lists of nearest neighbor edges.
    """
    cut_sim_graph = extract_failed_edge_subgraph(graph, edge_failed, similarity_measure)

    knn_dict = {}
    for edge in edge_failed:
        neighbors = cut_sim_graph[edge]
        sorted_neighbors = sorted(
            neighbors.items(), key=lambda x: x[1]["weight"], reverse=True
        )
        knn_dict[edge] = [nbr[0] for nbr in sorted_neighbors[:nearest_neighbors]]

    return knn_dict


def knn_failure_subgraphs_second_order(
    graph: _nx.Graph,
    edge_failed: list,
    similarity_measure: str = "predictor",
    nearest_neighbors: int = 8,
):
    """Get for each failed edge the k-nearest neighbors
    based on the collectivity measure, and then get the k-nearest neighbors
    of those neighbors as well (second order neighbors).
    Args:
        graph (_nx.Graph): The input graph to be clustered.
        edge_failed (list): The list of failed edges to be considered as nodes.
        similarity_measure (str, optional): The similarity measure to be used. Defaults to "predictor".
        nearest_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 8.
    Returns:
        knn_dict (dict): A dictionary where keys are failed edges and values are lists of nearest neighbor edges.
    """
    cut_sim_graph = extract_failed_edge_subgraph(graph, edge_failed, similarity_measure)

    knn_dict = {}
    for edge1, edge2 in it.combinations(edge_failed, 2):
        neighbors1 = cut_sim_graph[edge1]
        neighbors2 = cut_sim_graph[edge2]

        merged_neighbors = {}
        for nbr, attr in {**neighbors1, **neighbors2}.items():
            weight1 = neighbors1.get(nbr, {}).get("weight", 0)
            weight2 = neighbors2.get(nbr, {}).get("weight", 0)
            merged_neighbors[nbr] = max(
                weight1, weight2
            )  # Take the maximum weight if neighbor appears in both

        top_k = sorted(merged_neighbors.items(), key=lambda x: x[1], reverse=True)[
            :nearest_neighbors
        ]
        knn_dict[(edge1, edge2)] = [nbr[0] for nbr in top_k]
    return knn_dict


def get_list_highest_collectivity_edges(sim_graph: _nx.Graph) -> tuple[list, list]:
    """Take the edges of the similiarity graph and return the edges
    in order of highest to lowest collectivity measure."""

    nodes_sim_graph = list(sim_graph.nodes())
    degree_sim_graph = dict(_nx.degree(sim_graph, weight="weight"))

    sum_similarity = [degree_sim_graph.get(edge, 0) for edge in nodes_sim_graph]

    return nodes_sim_graph, sum_similarity
