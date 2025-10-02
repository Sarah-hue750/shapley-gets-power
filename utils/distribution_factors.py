#!/usr/bin/python3 
# -*- coding: utf-8 -*

"""Calculate the distribution factors for a given topology, i.e., graph."""

import numpy as _np
import networkx as _nx

from utils.data_handling import construct_incidencematrix_from_orientation,\
    construct_incidencematrix_from_orientation_multigraph

def calculate_PTDF_matrix(graph: _nx.Graph,
                          is_multigraph: bool = False):
    """
    Calculate the PTDF (Power Transfer Distribution Factor) matrix.
    The PTDF matrix is used to analyze how power flows change in response to changes in generation or load.
    """
    
    BB = _nx.laplacian_matrix(graph, weight='weight').todense()
    
    if is_multigraph:
        II = construct_incidencematrix_from_orientation_multigraph(graph, 
                                                        return_np_array=True)
    else:
        II = construct_incidencematrix_from_orientation(graph, 
                                                        return_np_array=True)

    BB_edge_list = [weight for uu, kk, weight 
                    in graph.edges(data='weight')]
    
    BB_d = -_np.diag(BB_edge_list)
    
    try:
        # Implicit matrix inversion
        BB_inv_n_II = _np.linalg.solve(BB, II)
        ptdf_matrix = _np.linalg.multi_dot((BB_d, II.T, BB_inv_n_II))
        
    except _np.linalg.LinAlgError:
        BB_inv = _np.linalg.pinv(BB)
        ptdf_matrix = _np.linalg.multi_dot((BB_d, II.T, BB_inv, II))
    
    return ptdf_matrix


def calculate_LODF_matrix(graph: _nx.Graph,
                          is_multigraph: bool = False):
    """
    Calculate the LODF matrix from the PTDF matrix.
    The LODF matrix is used to analyze the impact of line outages on power flows.
    """
    
    assert isinstance(graph, _nx.Graph), "Input must be a NetworkX graph"
    
    assert is_multigraph is graph.is_multigraph(), \
        "is_multigraph flag must match the graph type"
    
    ptdf_matrix = calculate_PTDF_matrix(graph,
                                        is_multigraph=is_multigraph)
    
    lodf_matrix = ptdf_matrix / (1 - _np.diag(ptdf_matrix))
    
    _np.fill_diagonal(lodf_matrix, -1) 
    
    # LODF for brides are not well defined, so they are set to nan.
    edge_list = list(graph.edges())
    bridges = _nx.bridges(graph)
    bridge_idx_ls = [edge_list.index(kk) for kk in bridges]
    for idx in bridge_idx_ls:
        lodf_matrix[:, idx] = _np.nan
    
    return lodf_matrix
