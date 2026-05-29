#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Calculate the distribution factors for a given topology, i.e., graph."""

import numpy as _np
import networkx as _nx

from utils.data_handling import (
    construct_incidencematrix_from_orientation,
    construct_incidencematrix_from_orientation_multigraph,
)


def calculate_PTDF_matrix(graph: _nx.graph, is_multigraph: bool = False) -> _np.ndarray:
    """Calulate the PTDF matrix"""

    # BB = _nx.laplacian_matrix(graph, weight='weight').todense()

    if is_multigraph:
        II = construct_incidencematrix_from_orientation_multigraph(
            graph, return_np_array=True
        )
    else:
        II = construct_incidencematrix_from_orientation(graph, return_np_array=True)

    BB_edge_list = [weight for _, _, weight in graph.edges(data="weight")]

    BB_d = _np.diag(BB_edge_list)

    BB = II @ BB_d @ II.T

    BB_inv = _np.linalg.pinv(BB)
    pdtf_matrix = _np.linalg.multi_dot((BB_d, II.T, BB_inv))

    return pdtf_matrix


def calculate_PTDF_matrix_times_incidence(
    graph: _nx.Graph, is_multigraph: bool = False
) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Calculate the PTDF (Power Transfer Distribution Factor) matrix.
    The PTDF matrix is used to analyze how power flows change in response to changes in generation or load.
    Note, this Matrix is already multiplied with the incidence matrix from the right, which
    assumes that power is injected at the from-node and withdrawn at the to-node of the corresponding line.

    Returns: Node-edge Incidence matrix and matrix with PTDF times incidence as numpy.ndarray
    """

    if graph.is_multigraph() is not is_multigraph:
        raise RuntimeError(
            "The choice of 'is_multigraph' " + "is not correct for the given graph"
        )

    # BB = _nx.laplacian_matrix(graph, weight='weight').todense()

    if is_multigraph:
        II = construct_incidencematrix_from_orientation_multigraph(
            graph, return_np_array=True
        )
    else:
        II = construct_incidencematrix_from_orientation(graph, return_np_array=True)

    BB_edge_list = [weight for _, _, weight in graph.edges(data="weight")]

    BB_d = _np.diag(BB_edge_list)
    BB = II @ BB_d @ II.T

    if not (abs(BB - _nx.laplacian_matrix(graph, weight="weight")) < 1e-8).all():
        raise RuntimeError("Laplacian is not calculated correctly.")

    try:
        # Implicit matrix inversion
        BB_inv_n_II = _np.linalg.solve(BB, II)
        ptdf_matrix_times_II = _np.linalg.multi_dot((BB_d, II.T, BB_inv_n_II))

    except _np.linalg.LinAlgError:
        BB_inv = _np.linalg.pinv(BB)
        ptdf_matrix_times_II = _np.linalg.multi_dot((BB_d, II.T, BB_inv, II))

    return II, ptdf_matrix_times_II


def calculate_LODF_matrix(
    graph: _nx.Graph,
    is_multigraph: bool = False,
    atol=1e-12,
    check_KCL: bool = True,
    verbose: bool = False,
):
    """
    Calculate the LODF matrix from the PTDF matrix.
    The LODF matrix is used to analyze the impact of line outages on power flows.
    """

    if not isinstance(graph, _nx.Graph):
        raise TypeError("Input must be a NetworkX graph")

    if not is_multigraph is graph.is_multigraph():
        raise RuntimeError("is_multigraph flag must match the graph type")

    II, ptdf_times_II_matrix = calculate_PTDF_matrix_times_incidence(
        graph, is_multigraph=is_multigraph
    )

    denom_lodf = 1 - _np.diag(ptdf_times_II_matrix)
    with _np.errstate(divide="ignore", invalid="ignore"):
        lodf_matrix = ptdf_times_II_matrix / denom_lodf

    _np.fill_diagonal(lodf_matrix, -1)

    is_close_zero_denom = abs(denom_lodf) < atol

    # LODF for brides are not well defined, so they are set to nan.
    edge_list = list(graph.edges())
    bridges = _nx.bridges(graph)
    bridge_idx_ls = [edge_list.index(kk) for kk in bridges]
    non_bridge_idx_ls = set(range(len(edge_list))) - set(bridge_idx_ls)

    if not set(bridge_idx_ls) == set(_np.where(is_close_zero_denom)[0]):
        raise RuntimeError("Bridges should be points with PTDF_times_II being 1.")

    for idx in bridge_idx_ls:
        lodf_matrix[:, idx] = _np.nan

    if verbose:
        print(f"Set {len(bridge_idx_ls)} col to nan due to bridges!")

    if check_KCL:
        if not all([sum(II @ lodf_matrix[:, kk]) < atol for kk in non_bridge_idx_ls]):
            raise RuntimeError("KCL is not obeyed! Are there any bridges or dead-ends?")

    return lodf_matrix
