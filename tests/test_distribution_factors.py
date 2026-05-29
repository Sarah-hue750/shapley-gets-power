#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the calculation of LODF"""

import numpy as _np
import networkx as _nx

import random

from scipy.sparse import csr_matrix

from utils import distribution_factors
from utils.cascade_simulation import solve_lpf
from utils.data_handling import construct_incidencematrix_from_orientation, construct_incidencematrix_from_orientation_multigraph

from matplotlib import pyplot as plt

def test_lodf_square_grid(len_grid: int = 8, 
                          edge_weights: float | list[float] = .3,
                          atol: float = 1e-13):
    """Test lodf calculation for square grid."""

    graph_square = _nx.grid_2d_graph(len_grid, len_grid)
    
    orientation_dict = {xx: (xx[0], xx[1]) for xx in graph_square.edges()}
    
    if isinstance(edge_weights, list):
        edge_weights = {kk: edge_weights[idx_kk] for idx_kk, kk in enumerate(graph_square.edges())}
    _nx.set_edge_attributes(graph_square, edge_weights, 'weight')
    _nx.set_edge_attributes(graph_square, orientation_dict, 'orientation')
    
    # Calculate LODFs
    lodf_matrix = distribution_factors.calculate_LODF_matrix(graph_square, 
                                               is_multigraph=False,
                                               check_KCL=True)
    
    II = construct_incidencematrix_from_orientation(graph_square, 
                                                    return_np_array=True)
    
    assert all([sum(II @ lodf_matrix[:, kk]) < atol
                    for kk in range(lodf_matrix.shape[0])]), \
                        "KCL is not obeyed! Are there any bridges or dead-ends?"
    
    return
    
    
def test_lodf_double_square_grid(edge_weights: float | list[float] = .3, 
                                 connection_weight: float = 1.,
                                 len_grid: int = 20,
                                 atol: float = 1e-13):
    """Test lodf calculation for two square grids that are connected 
    in the middle."""
    
    # Setup dual graph
    graph_left = _nx.grid_2d_graph(len_grid, len_grid)
    pos_dict_left = {f"left_{xx}": xx for xx in graph_left.nodes()}
    graph_left = _nx.relabel_nodes(graph_left, lambda xx: f'left_{xx}')
    
    graph_right = _nx.grid_2d_graph(len_grid, len_grid)
    pos_dict_right = {f"right_{xx}": (xx[0] + len_grid + 0.5, xx[1]) 
                      for xx in graph_right.nodes()}
    graph_right = _nx.relabel_nodes(graph_right, lambda xx: f'right_{xx}')
    
    graph = _nx.union(graph_left, graph_right)

    ## Set parameters (edge_weight => positive susceptance)
    if isinstance(edge_weights, list):
        edge_weights = {kk: edge_weights[idx_kk] 
                        for idx_kk, kk in enumerate(graph.edges())}
    _nx.set_edge_attributes(graph, edge_weights, 'weight')
    
    # Connect the two grids
    middle_coord = len_grid // 2
    graph.add_edge(f"left_({len_grid-1}, {middle_coord})", 
                   f"right_({0}, {middle_coord})", 
                   weight=connection_weight)
    graph.add_edge(f"left_({len_grid-1}, {middle_coord-1})", 
                   f"right_({0}, {middle_coord-1})", 
                   weight=connection_weight)
    
    orientation_dict = {edges: (edges[0], edges[1]) for edges in graph.edges()}
    _nx.set_edge_attributes(graph, orientation_dict, 'orientation')
    
    pos_total = {**pos_dict_left, **pos_dict_right}
    
    # LODF
    lodf_matrix = distribution_factors.calculate_LODF_matrix(graph, 
                                                      is_multigraph=False)
    
    
    II = construct_incidencematrix_from_orientation(graph, return_np_array=True)
    
    assert all([sum(II @ lodf_matrix[:, kk]) < atol
                    for kk in range(lodf_matrix.shape[0])]), \
                        "KCL is not obeyed! Are there any bridges or dead-ends?"
    
    return


def test_lodf_double_square_grid_random_bb(nn_tries: int = 5,
                                           val_range: tuple[float, float] = (0.3, 1.2)):
    """Test LODF calculation for random BB."""
    
    for _ in range(nn_tries):
        grid_len = 5
        nr_edges_single = 2*grid_len**2 - 2*grid_len
        nr_edges_double = 2*nr_edges_single
        
        rand_weights_single = [random.uniform(min(val_range), max(val_range)) for _ in range(nr_edges_single)]
        rand_weights_double = [random.uniform(min(val_range), max(val_range)) for _ in range(nr_edges_double)]
        
        
        test_lodf_square_grid(len_grid=grid_len, 
                              edge_weights=rand_weights_single)
        
        test_lodf_double_square_grid(len_grid=grid_len,
                                     edge_weights=rand_weights_double)
        

def calc_lodf_numerically(graph: _nx.Graph, 
                          P_vec: _np.ndarray):
    """Calculate the lodf numerically"""
    
    assert _nx.is_connected(graph) and len(list(_nx.bridges(graph))) == 0
    
    is_multigraph = graph.is_multigraph()
    
    nr_edges = graph.number_of_edges()
    BB_d = _np.diag([ww for _, _, ww in graph.edges(data='weight')])
    
    if is_multigraph:
        II = construct_incidencematrix_from_orientation_multigraph(graph)
        edge_list = list(graph.edges(keys=True))
    else:
        II = construct_incidencematrix_from_orientation(graph)
        edge_list = list(graph.edges())
        
    LL = csr_matrix(II @ BB_d @ II.T)
        
    lpf_0 = solve_lpf(P_vec, BB_d, II, LL)
    
    
    lodf_numerical = _np.full((nr_edges, nr_edges), _np.nan, float)
    
    
    for idx_r, id_edge in enumerate(edge_list):
        graph_r = graph.copy()
        
        if is_multigraph:
            graph_r.remove_edge(id_edge[0], id_edge[1], key=id_edge[2])
        else:
            graph_r.remove_edge(id_edge[0], id_edge[1])
        
        if is_multigraph:
            II_r = construct_incidencematrix_from_orientation_multigraph(graph_r)
        else:
            II_r = construct_incidencematrix_from_orientation(graph_r)
        BB_d_r = _np.diag([ww for _, _, ww in graph_r.edges(data='weight')])
        LL_r = csr_matrix(II_r @ BB_d_r @ II_r.T)
        
        lpf_r = solve_lpf(P_vec, BB_d_r, II_r, L=LL_r)

        mask_r = [True] * nr_edges
        mask_r[idx_r] = False
        
        col_lodf =  (lpf_r - lpf_0[mask_r])/lpf_0[idx_r]

        lodf_numerical[mask_r, idx_r] = col_lodf
    
    _np.fill_diagonal(lodf_numerical, -1)
    
    return lodf_numerical

def test_compare_lodf_and_numerical_lodf(P_i_range=(.1, .3), 
                                         edge_weight_range=(.5, .7),
                                         nr_tries: int = 20):
    """Compare the LODF and a numerical derived version of the LODFs"""
    
    # Solve power flows:
    ## Setup
    for _ in range(nr_tries):
        # Define system for squared
        len_grid = 5
        graph_square = _nx.grid_2d_graph(len_grid, len_grid)
        nr_edges = len(graph_square.edges())
        
        orientation_dict = {xx: (xx[0], xx[1]) for xx in graph_square.edges()}
        
        edge_weights = {kk: random.uniform(min(edge_weight_range), max(edge_weight_range)) 
                        for kk in graph_square.edges()}
        _nx.set_edge_attributes(graph_square, edge_weights, 'weight')
        _nx.set_edge_attributes(graph_square, orientation_dict, 'orientation')
        
        pos_dict = {xx: (xx[0], xx[1]) for xx in graph_square.nodes()}
        
        P_i = random.uniform(min(P_i_range), max(P_i_range))
        PP_vec = _np.zeros(len(graph_square))
        PP_vec[4] = P_i
        PP_vec[20] = - P_i
        
        assert abs(sum(PP_vec)) < 1e-10, "System not balanced!"
        
        lodf_numerical = calc_lodf_numerically(graph_square, PP_vec)
        
        # Analytical LODF_matrix
        lodf_matrix = distribution_factors.calculate_LODF_matrix(graph_square)
        
        # Compare
        assert ((lodf_matrix - lodf_numerical) < 1e-13).all(), "LODF numerical example and analytical don't agree!"
        
    return


def test_lodf_diamond_graph_numerical(weight_range=(.4, .6), Pi_range=(.1, .3),
                            nr_tries: int = 10):
    """Test for a simple diamond
    """
    
    # Graph
    for _ in range(nr_tries):
        graph = _nx.Graph()
        
        graph.add_node(0)
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        
        graph.add_edge(0, 1, 
                       weight=random.uniform(min(weight_range), max(weight_range)))
        graph.add_edge(0, 2, 
                       weight=random.uniform(min(weight_range), max(weight_range)))
        graph.add_edge(0, 3, 
                       weight=random.uniform(min(weight_range), max(weight_range)))
        graph.add_edge(1, 2, 
                       weight=random.uniform(min(weight_range), max(weight_range)))
        graph.add_edge(2, 3, 
                       weight=random.uniform(min(weight_range), max(weight_range)))
        
        nr_edges = graph.number_of_edges()
        orientation_dict = {xx: (xx[0], xx[1]) for xx in graph.edges()}
        _nx.set_edge_attributes(graph, orientation_dict, "orientation")
        
        Pi = random.uniform(min(Pi_range), max(Pi_range))
        PP_vec = _np.zeros(len(graph))
        PP_vec[0] = Pi
        PP_vec[-1] = Pi
        
        lodf_numerical = calc_lodf_numerically(graph, PP_vec)
        
        lodf_matrix = distribution_factors.calculate_LODF_matrix(graph)
        
        assert (abs(lodf_matrix - lodf_numerical) < 1e-13).all(), "LODF numerical and analytical should be the same"
    
    return 

def test_lodf_one_bridge(len_grid=5, edge_weights_range=(.3, .5),
                         connection_weight_range=(.3, .5), nr_tries=10):
    """Test a case with one bridge between two parts and a dead end.
    """
    
    # Setup system
    for _ in range(nr_tries):
        graph_left = _nx.grid_2d_graph(len_grid, len_grid)
        pos_dict_left = {f"left_{xx}": xx for xx in graph_left.nodes()}
        graph_left = _nx.relabel_nodes(graph_left, lambda xx: f'left_{xx}')
        
        graph_right = _nx.grid_2d_graph(len_grid, len_grid)
        pos_dict_right = {f"right_{xx}": (xx[0] + len_grid + 0.5, xx[1]) 
                        for xx in graph_right.nodes()}
        graph_right = _nx.relabel_nodes(graph_right, lambda xx: f'right_{xx}')
        
        graph = _nx.union(graph_left, graph_right)

        ## Set parameters (edge_weight => positive susceptance)
        edge_weights = {edge: random.uniform(min(edge_weights_range), max(edge_weights_range))
                        for edge in graph.edges()}
        _nx.set_edge_attributes(graph, edge_weights, 'weight')
        
        # Connect the two grids
        middle_coord = len_grid // 2
        connection_nodes =  (f"left_({len_grid-1}, {middle_coord})", 
                            f"right_({0}, {middle_coord})")
        graph.add_edge(f"left_({len_grid-1}, {middle_coord})", 
                    f"right_({0}, {middle_coord})", 
                    weight=random.uniform(min(connection_weight_range), 
                                          max(connection_weight_range)))
        connection_edge_idx = list(graph.edges()).index(connection_nodes)
        
        orientation_dict = {edges: (edges[0], edges[1]) for edges in graph.edges()}
        _nx.set_edge_attributes(graph, orientation_dict, 'orientation')
        
        lodf_matrix = distribution_factors.calculate_LODF_matrix(graph, is_multigraph=False)
        
        assert _np.isnan(lodf_matrix[:, connection_edge_idx]).all()
    
    return


def test_lodf_multigraph_numerical(Pi_range=(.3, .4),
                                   edge_weight_range=(.5, .8)):
    """Check if simulations lpf lodf calculation agrees with LODF"""
    
    graph = _nx.MultiGraph()
    
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1, weight=random.uniform(min(edge_weight_range), 
                                               max(edge_weight_range)))
    graph.add_edge(0, 1,weight=random.uniform(min(edge_weight_range), 
                                               max(edge_weight_range)))
    
    graph.add_edge(1, 2, weight=random.uniform(min(edge_weight_range), 
                                               max(edge_weight_range)))
    graph.add_edge(0, 2, weight=random.uniform(min(edge_weight_range), 
                                               max(edge_weight_range)))
 
    orientation_dict = {edges: (edges[0], edges[1]) for edges in graph.edges(keys=True)}
    _nx.set_edge_attributes(graph, orientation_dict, 'orientation')
    nr_edges = graph.number_of_edges()
    
    Pi = random.uniform(min(Pi_range), max(Pi_range)
                        )
    P_vec = _np.zeros(len(graph))
    P_vec[0] = Pi
    P_vec[1] = -Pi
    
    
    lodf_numerical = calc_lodf_numerically(graph, P_vec)
    
    lodf_analytical = distribution_factors.calculate_LODF_matrix(graph, 
                                                                 is_multigraph=True)
    
    assert (abs(lodf_numerical - lodf_analytical) < 1e-13).all()
    
    return
