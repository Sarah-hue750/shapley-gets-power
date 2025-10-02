#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preparation and conversion of data for system split simulation and evaluation
"""

import os
import networkx as nx
import numpy as np
import random
import pypsa
from scipy import sparse
import copy
import csv



def serialize_edge(edge):
    return f"{edge[0]}-{edge[1]}-{edge[2]}"


def deserialize_edge(edge_str):
    u, v, k = edge_str.split("-")
    return ((u), (v), int(k))


def serialize_edges(edges):
    return ";".join(serialize_edge(e) for e in edges)


def deserialize_edges(edges_str):
    if not edges_str:
        return []
    return [deserialize_edge(e) for e in edges_str.split(";")]


def load_cases_csv(filename=""):
    # error if file does not exist

    import os

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")
    cases = []
    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                e1 = deserialize_edge(row["edge1"])
                e2 = deserialize_edge(row["edge2"])
                reversed_edges = deserialize_edges(row["reversed_edges"])
                cases.append((e1, e2, reversed_edges))
    except FileNotFoundError:
        pass
    return cases

def build_networkx_graph(pypsa_network, snet_index=None):
    """Build a networkx graph from the pypsa networks"""
    pypsa_network.determine_network_topology()

    try:
        snet = pypsa_network.sub_networks["obj"][snet_index]
    except KeyError:
        snet = pypsa_network

    branches = snet.branches()
    positions = pypsa_network.buses[["x", "y"]]
    pos = dict(zip(positions.index, list(zip(positions.x, positions.y))))

    branches = branches[
        ["bus0", "bus1", "x_pu_eff", "s_nom", "num_parallel", "v_nom", "i_nom"]
    ]

    F = nx.Graph()
    for line_index, line in branches.iterrows():
        if not F.has_edge(line["bus0"], line["bus1"]):
            F.add_edge(
                line["bus0"],
                line["bus1"],
                weight=1 / line["x_pu_eff"],
                orientation=(line["bus0"], line["bus1"]),
                line_index=[line_index[1]],
                s_nom=line["s_nom"],  # MVA
                num_parallel=line["num_parallel"],
                v_nom=line["v_nom"],  # MV
                i_nom=line["i_nom"],  # kA
            )
        else:
            raise (RuntimeError("There duplicated edges in the PyPSA network"))

    nx.set_node_attributes(F, pos, "pos")
    return F


def construct_incidencematrix_from_orientation(Graph, return_np_array=True):
    """Construct incidence matrix for a graph with edge keyword orientation specifying the edge order
    NOTE: This function was tweaked based on networkx incidence matrix function
    https://networkx.org/documentation/stable/_modules/networkx/linalg/graphmatrix.html#incidence_matrix
    """

    edgelist = list(Graph.edges())
    nodelist = list(Graph.nodes())

    node_index = {node: i for i, node in enumerate(nodelist)}
    B = sparse.lil_matrix((len(nodelist), len(edgelist)))
    orientations = nx.get_edge_attributes(Graph, "orientation")
    assert bool(orientations), "Graph edges must have an 'orientation' attribute"

    for ii, edge in enumerate(edgelist):
        (uu, vv) = orientations[edge]

        n1 = node_index[uu]
        # TODO why should this be float and not int?
        B[n1, ii] = 1.0
        n2 = node_index[vv]
        B[n2, ii] = -1.0

    if return_np_array:
        return_val = B.toarray()
    else:
        return_val = B.asformat("csr")

    return return_val


def get_effective_injections(network, snapshot, nx_graph):
    """Get effective nodal injections on nx_graph for certain snapshot from PyPSA network.

    Args:
        network (pypsa.network): PyPSA network
        snapshot (string): Format '%Y-%m-%d %H:00'
        nx_graph (networkx graph): Graph pf power system under investigation.

    Returns:
        numpy array: n_nodes x 1 array with nodal injections
    """

    I_m = construct_incidencematrix_from_orientation(nx_graph)

    flows_network = network.lines_t.p0.loc[snapshot]
    flows_matrix = np.array(
        [
            flows_network[attribs["line_index"][0]]
            for u, v, attribs in nx_graph.edges(data=True)
        ]
    )

    P0 = np.dot(I_m, flows_matrix)

    return P0


def nx_edges_to_matrix_indices(nx_edges, nx_graph):
    """
    Transform edge names from networkx graph format to matrix format.
    """

    lookup_dict = dict(zip(nx_graph.edges(), range(nx_graph.number_of_edges())))
    matrix_indices = [lookup_dict[link_name] for link_name in nx_edges]

    return matrix_indices


def matrix_indices_to_nx_edges(indices, nx_graph):
    """
    Transform edge indices from matrix format to edge names in networkx graph.
    """

    lookup_dict = dict(
        zip(range(nx_graph.number_of_edges()), nx_graph.edges(keys=True))
    )
    edge_names = [lookup_dict[ind] for ind in indices]

    return edge_names


def get_matrices_from_nx_graph(nx_graph):
    """
    Extract incidence matrix, susceptance matrix, effective number of parallel lines per edge
    and line limits per edge from networkx graph.
    """

    # Build incidence matrix and susceptance matrix
    I_m = construct_incidencematrix_from_orientation(nx_graph, return_np_array=False)
    B_d = sparse.spdiags(
        np.array([attribs["weight"] for u, v, attribs in nx_graph.edges(data=True)]),
        0,
        nx_graph.number_of_edges(),
        nx_graph.number_of_edges(),
    ).asformat("csr")

    # Get array of num_parallels and of line limits
    num_parallels = np.array(
        [attribs["num_parallel"] for u, v, attribs in nx_graph.edges(data=True)]
    )
    line_limits = np.array(
        [attribs["s_nom"] for u, v, attribs in nx_graph.edges(data=True)]
    )

    return I_m, B_d, num_parallels, line_limits


def shuffle_knn_dict(original_knn_dict, seed):
    """
    Creates a random knn dict that matches the number of neirest neighbours of another dict and just shuffles the lines.
    Args:
        original_knn_dict (dict): Knn dict to shuffle with line as key and neirest neighbours as values.
        seed (int): random seed for reproducability
    Returns:
        rand_knn_dict (dict):  randomly shuffled knn dict.
    """
    random.seed(seed)
    rand_knn_dict = {}
    keys = list(original_knn_dict.keys())

    for key, neighbours in original_knn_dict.items():
        n_neighbors = len(neighbours)
        possible_neighbors = list(set(keys) - {key})
        random_neighbors = random.sample(possible_neighbors, n_neighbors)
        rand_knn_dict[key] = random_neighbors
    return rand_knn_dict


def shuffle_knn_dict_second_order(original_knn_dict, seed):
    """
    Creates a random knn dict that matches the number of neirest neighbours of another dict and just shuffles the lines.
    Args:
        original_knn_dict (dict): Knn dict to shuffle with line as key and neirest neighbours as values.
        seed (int): random seed for reproducability
    Returns:
        rand_knn_dict (dict):  randomly shuffled knn dict.
    """
    random.seed(seed)
    rand_knn_dict = {}
    keys = list(original_knn_dict.keys())

    for key, neighbours in original_knn_dict.items():
        n_neighbors = len(neighbours)
        possible_neighbors = list(set(keys) - set(key))
        random_neighbors = random.sample(possible_neighbors, n_neighbors)
        rand_knn_dict[key] = random_neighbors
    return rand_knn_dict


def get_knn_dict(original_knn_dict, G):
    """
    Creates a cluster dict from a knn dict.
    Args:
        original_knn_dict (dict): Knn dict to convert, with cluster indices as values and lines as index.
        G (networkx.Graph): The original graph.
    Returns:
        knn_cluster_dict (dict): Cluster dict with matrix indeces.
    """
    knn_cluster_dict = {}
    for key, value in original_knn_dict.items():
        edges_mx = nx_edges_to_matrix_indices(list(value), G)
        key_mx = nx_edges_to_matrix_indices([key], G)[0]
        knn_cluster_dict[key_mx] = edges_mx
    return knn_cluster_dict


def get_knn_dict_second_order(original_knn_dict, G):
    """
    Creates a cluster dict from a knn dict.
    Args:
        original_knn_dict (dict): Knn dict to convert, with cluster indices as values and lines as index.
        G (networkx.Graph): The original graph.
    Returns:
        knn_cluster_dict (dict): Cluster dict with matrix indeces.
    """
    knn_cluster_dict = {}
    for key, value in original_knn_dict.items():
        edges_mx = nx_edges_to_matrix_indices(list(value), G)
        key_mx = nx_edges_to_matrix_indices(list(key), G)
        knn_cluster_dict[tuple(key_mx)] = edges_mx
    return knn_cluster_dict


##### multigraph functions #####


def construct_incidencematrix_from_orientation_multigraph(Graph, return_np_array=True):
    """
    Construct incidence matrix for a MultiGraph or MultiDiGraph with edge attribute
    'orientation' = (u, v), specifying direction of each edge.

    Args:
        Graph (networkx.MultiGraph or MultiDiGraph): Input graph.
        return_np_array (bool): Return as NumPy array if True, else as scipy CSR matrix.

    Returns:
        np.ndarray or scipy.sparse.csr_matrix: Incidence matrix (nodes × edges).
    """

    # For MultiGraphs: edges have keys
    edgelist = list(Graph.edges(keys=True))
    nodelist = list(Graph.nodes())

    node_index = {node: i for i, node in enumerate(nodelist)}
    B = sparse.lil_matrix((len(nodelist), len(edgelist)))
    orientations = nx.get_edge_attributes(Graph, "orientation")

    for ii, edge in enumerate(edgelist):
        uu, vv, k = edge
        orientation = orientations.get(edge)

        if orientation is None:
            raise ValueError(f"Missing 'orientation' attribute for edge {edge}.")

        src, tgt = orientation
        B[node_index[src], ii] = 1.0
        B[node_index[tgt], ii] = -1.0

    if return_np_array:
        return B.toarray()
    else:
        return B.asformat("csr")


def get_effective_injections_multigraph(network, snapshot, nx_graph):
    """Get effective nodal injections on nx_graph for certain snapshot from PyPSA network.

    Args:
        network (pypsa.network): PyPSA network
        snapshot (string): Format '%Y-%m-%d %H:00'
        nx_graph (networkx graph): Graph pf power system under investigation.

    Returns:
        numpy array: n_nodes x 1 array with nodal injections
    """

    I_m = construct_incidencematrix_from_orientation_multigraph(nx_graph)

    flows_network = network.lines_t.p0.loc[snapshot]
    flows_matrix = np.array(
        [
            flows_network[attribs["line_index"][0]]
            for u, v, attribs in nx_graph.edges(data=True)
        ]
    )

    P0 = np.dot(I_m, flows_matrix)

    return P0


def get_matrices_from_nx_graph_multigraph(nx_graph):
    """
    Extract incidence matrix, susceptance matrix, effective number of parallel lines per edge
    and line limits per edge from networkx graph.
    """

    # Build incidence matrix and susceptance matrix
    I_m = construct_incidencematrix_from_orientation_multigraph(
        nx_graph, return_np_array=False
    )
    B_d = sparse.spdiags(
        np.array([attribs["weight"] for u, v, attribs in nx_graph.edges(data=True)]),
        0,
        nx_graph.number_of_edges(),
        nx_graph.number_of_edges(),
    ).asformat("csr")

    # Get array of num_parallels and of line limits
    num_parallels = np.array(
        [attribs["num_parallel"] for u, v, attribs in nx_graph.edges(data=True)]
    )
    line_limits = np.array(
        [attribs["s_nom"] for u, v, attribs in nx_graph.edges(data=True)]
    )

    return I_m, B_d, num_parallels, line_limits


from utils.cascade_simulation import calc_num_parallel_after_failure

def deaggregate_parallel_lines(G: nx.Graph):
    """
    Deaggregate parallel lines in a networkx graph based on the 'num_parallel' attribute.

    Parameters:
    -----------
    G : networkx.Graph
        The input graph with parallel lines.

    Returns:
    --------
    G_deaggregated : networkx.MultiGraph
        A new graph with deaggregated lines, where each parallel line is represented as a separate edge with scaled attributes.
    """

    # create a new MultiGraph to hold the deaggregated edges
    G_deaggregated = nx.MultiGraph()  # Create a copy of the original graph
    # include all nodes from the original graph
    G_deaggregated.add_nodes_from(G.nodes(data=True))

    # Initialize a new list to hold the new edges
    new_edges = []

    for u, v, data in G.edges(data=True):
        num_parallel = data.get("num_parallel", 1)
        susceptance = data.get("weight", 1)
        line_limit = data.get("s_nom", 1)
        v_nom = data.get("v_nom")
        i_nom = data.get("i_nom")
        new_num_parallel_before_removal = (
            num_parallel  # Create a copy to avoid modifying the original
        )

        while new_num_parallel_before_removal > 0:
            new_num_parallel_after_removal = calc_num_parallel_after_failure(
                num_parallel=new_num_parallel_before_removal
            )

            d_num_parallel = (
                new_num_parallel_before_removal - new_num_parallel_after_removal
            )

            # calculate network parameters accordingly
            num_par_factor = d_num_parallel / num_parallel
            num_parallel = d_num_parallel
            subline_susceptance = susceptance * num_par_factor
            subline_line_limit = line_limit * num_par_factor

            # create new edge data
            new_edge_data = data.copy()
            new_edge_data["num_parallel"] = num_parallel
            new_edge_data["weight"] = subline_susceptance
            new_edge_data["s_nom"] = subline_line_limit
            new_edge_data["v_nom"] = v_nom
            new_edge_data["i_nom"] = i_nom * num_par_factor

            # add new edge to the list
            new_edges.append((u, v, new_edge_data))

            new_num_parallel_before_removal = new_num_parallel_after_removal

    G_deaggregated.add_edges_from(new_edges)
    return G_deaggregated


def nx_edges_to_matrix_indices(nx_edges, nx_graph):
    """
    Transform edge names from a MultiGraph/MultiDiGraph to matrix indices.
    Assumes nx_edges are (u, v, key) and that matrix rows follow edge order.
    """
    # Include keys in the lookup
    lookup_dict = {edge: i for i, edge in enumerate(nx_graph.edges(keys=True))}
    matrix_indices = [lookup_dict[edge] for edge in nx_edges]

    return matrix_indices


def add_flows_to_graph(G_multi, flows_0):
    """
    Adds flow, current and load values to the edges of the graph G_multi.
    """
    G_0 = G_multi.copy()
    # add flows to graph
    exs = list(G_0.edges(keys=True))
    mxs = nx_edges_to_matrix_indices(G_0.edges(keys=True), G_0)
    for ex, mx in zip(exs, mxs):
        data = G_0.edges[ex]
        # get original flow
        if "flow" not in data:
            # add flow
            data["flow"] = flows_0[mx]
            data["del_flow"] = None
            # add load
            if data["s_nom"] != 0:
                data["load"] = np.abs(data["flow"]) / data["s_nom"]
            else:
                data["load"] = float("inf")
            # add current
            if data["v_nom"] != 0:
                data["current"] = (data["flow"]) / (np.sqrt(3) * data["v_nom"])
            else:
                data["current"] = float("inf")

        elif "del_flow" in G_multi.edges[ex]:
            data["del_flow"] = (
                flows_0[mx] - G_multi.edges[ex]["flow"]
            )  # Could this be that this should be like this?
            data["flow"] = flows_0[mx]

    return G_0


from utils.cascade_simulation import solve_lpf


def get_flownetwork_without_lines(
    G: nx.Graph, P: np.ndarray, lines_to_remove: list
) -> nx.Graph:
    """
    Returns a new graph with specified lines removed.
    """

    G_copy = G.copy()

    I_m_orig, B_d_orig, num_parallels_orig, line_limitsorig = (
        get_matrices_from_nx_graph_multigraph(G)
    )

    mxs = nx_edges_to_matrix_indices(
        lines_to_remove, G
    )  # translate lines to matrix indices

    # Remove the specified lines from the B_d matrix (0 supceptance)
    B_d_after = B_d_orig.copy()
    for mx in mxs:
        if mx < B_d_after.shape[0] and mx < B_d_after.shape[1]:
            B_d_after[mx, mx] = 0

        else:
            # error
            raise ValueError(
                f"Matrix index {mx} is out of bounds for B_d matrix of shape {B_d_after.shape}."
            )
    # Solve the linear power flow problem with the modified B_d matrix
    flows_after = solve_lpf(P=P, B_d=B_d_after, I=I_m_orig)
    # Add the flows to the graph
    G_copy = add_flows_to_graph(G_copy, flows_after)

    # calculate (over) load in percent

    nx.set_edge_attributes(
        G_copy,
        {
            (u, v, k): (
                np.abs(data["flow"]) / data["s_nom"]
                if data["s_nom"] != 0
                else float("inf")
            )
            for u, v, k, data in G_copy.edges(keys=True, data=True)
        },
        name="load",
    )

    # calculate current in A
    nx.set_edge_attributes(
        G_copy,
        {
            (u, v, k): (
                (data["flow"]) / (np.sqrt(3) * data["v_nom"])
                if data["s_nom"] != 0
                else float("inf")
            )
            for u, v, k, data in G_copy.edges(keys=True, data=True)
        },
        name="current",
    )
    return G_copy


def get_shap_network(G_0, G_0_small, shap, outage_lines):
    """
    Creates a graph that includes the shap values as properties of its edges.
    """
    for line in outage_lines:
        shap[line] = 0
    shap_nx_edges = matrix_indices_to_nx_edges(shap.keys(), G_0)
    shap_nx = {shap_nx_edges[idx]: v for idx, (k, v) in enumerate(shap.items())}

    G_0_small_copy = G_0_small.copy()
    for edge in list(shap_nx_edges):
        G_0_small_copy.edges[edge]["del_flow"] = shap_nx[edge]
        G_0_small_copy.edges[edge]["flow"] = (
            G_0_small_copy.edges[edge]["flow"] + shap_nx[edge]
        )
        G_0_small_copy.edges[edge]["current"] = (G_0_small_copy.edges[edge]["flow"]) / (
            np.sqrt(3) * G_0_small_copy.edges[edge]["v_nom"]
        )
        G_0_small_copy.edges[edge]["load"] = (
            np.abs(G_0_small_copy.edges[edge]["flow"])
            / G_0_small_copy.edges[edge]["s_nom"]
        )
    return G_0_small_copy


def sort_edge_indices(G):
    """
    Sorts the edge indices of a NetworkX graph in a consistent order.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    list: Sorted list of edge indices.
    """

    G_new = nx.MultiGraph()
    # add nodes
    G_new.add_nodes_from(G.nodes(data=True))

    # define a dictionary to hold the sorted edges
    for u, v, key, data in G.edges(keys=True, data=True):
        u, v = sorted([u, v])
        G_new.add_edge(u, v, key=key, **data)
    return G_new


def remove_random_edges(G, num_edges, seed=None):
    """
    Removes a specified number of random edges from a NetworkX graph.
    This function ensures that the graph remains connected after each edge removal.

    Parameters:
    G (networkx.Graph): The input graph.
    num_edges (int): The number of edges to remove.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    removed_edges (list): List of edges that were removed.
    """
    removed_edges = []
    G_copy = sort_edge_indices(G)
    if seed is not None:
        random.seed(seed)

    removed_count = 0
    edges = list(G_copy.edges(keys=True))
    random.shuffle(edges)

    for edge in edges:
        if removed_count >= num_edges:
            break
        G_copy.remove_edge(*edge)
        if nx.is_connected(G_copy):
            removed_count += 1
            removed_edges.append(edge)
        else:
            # If removing this edge disconnects the graph, add it back
            G_copy.add_edge(*edge)
    return removed_edges


def get_attributes_from_flow_change(G, flow_change_dict, attr):
    """
    Returns dict of edge attributes given the flow change.
    Args:
        G (networkx.Graph): The input graph.
        flow_change_dict (dict): Dict with matrix indices as keys of flow changes to perform on the graph.
        attr (string): Either "load" or "current" to indicate what attribute of edges is desired.
    Returns:
        attr_dict (dict): Dict with matrix indices as keys of specified attribute.
    """
    flow_change_nx_edges = matrix_indices_to_nx_edges(flow_change_dict.keys(), G)
    flow_change_nx = {
        flow_change_nx_edges[idx]: v
        for idx, (k, v) in enumerate(flow_change_dict.items())
    }

    G_copy = G.copy()
    for edge in list(flow_change_nx):
        G_copy.edges[edge]["del_flow"] = flow_change_nx[edge]
        G_copy.edges[edge]["flow"] = G_copy.edges[edge]["flow"] + flow_change_nx[edge]
        G_copy.edges[edge]["current"] = (G_copy.edges[edge]["flow"]) / (
            np.sqrt(3) * G_copy.edges[edge]["v_nom"]
        )
        G_copy.edges[edge]["load"] = (
            np.abs(G_copy.edges[edge]["flow"]) / G_copy.edges[edge]["s_nom"]
        )
    if attr == "nx_graph":
        return G_copy
    if attr not in ["load", "current"]:
        print("Unknown attribute.")
        return -1
    else:
        attr_dict = {}
        for matrix_idx, edge in zip(flow_change_dict.keys(), flow_change_nx.keys()):
            attr_dict[matrix_idx] = G_copy.edges[edge][attr]
        return attr_dict


def remove_negligable_affected_lines(
    results, attr_thresholds={"flow_change": 7, "load": 0.01, "current": 0.01}
):
    """
    Sets all abs_diff and rel_diff in a result dict to zero if the corresponding exact_shap is below a threshold for the respective attribute.
    Args:
        results(dict): dict with results,
        attr_threshold(dict): dict with threshold for each attribute.
    """

    results_copy = copy.deepcopy(results)
    for cluster_result in results_copy["cluster_results"]:
        for attr, threshold in attr_thresholds.items():
            values = {}
            for outage_line, shap_dict in results_copy[f"exact_shap_{attr}"].items():
                for affected_line, value in shap_dict.items():
                    values[affected_line] = values.get(affected_line, 0) + value
            for outage_line, shap_dict in results_copy[f"exact_shap_{attr}"].items():
                for affected_line, value in shap_dict.items():
                    if abs(values[affected_line]) < threshold:
                        cluster_result["differences"][f"abs_diff_{attr}"][outage_line][
                            affected_line
                        ] = None
                        cluster_result["differences"][f"rel_diff_{attr}"][outage_line][
                            affected_line
                        ] = None
    return results_copy


# def find_next_neighbors(G, edges):
#     next_neighbors = set()
#     for edge in edges:
#         u, v, k = edge
#         neighbors_u = set(G.neighbors(u))
#         neighbors_v = set(G.neighbors(v))

#         next_neighbors.update(neighbors_u)
#         next_neighbors.update(neighbors_v)

#     return next_neighbors


# def find_next_next_neighbors_until_connected(G, edges):

#     print(f"Finding next next neighbors for edges: {edges}")

#     # Step 1: collect neighbors of all given edges
#     all_nodes_neighbors = set()
#     for edge in edges:
#         nodes_neighbors = find_next_neighbors(G, [edge])
#         all_nodes_neighbors |= nodes_neighbors  # union

#     # Step 2: get edges connected to those neighbors
#     all_edges_neighbors = [
#         (u, v, k)
#         for u, v, k in G.edges
#         if u in all_nodes_neighbors or v in all_nodes_neighbors
#     ]

#     # Step 3: induced subgraph
#     G_u = G.subgraph(all_nodes_neighbors).copy()

#     # Step 4: expand until connected
#     while not nx.is_connected(G_u):
#         new_nodes_neighbors = set()
#         for e in all_edges_neighbors:
#             new_nodes_neighbors |= find_next_neighbors(G, [e])

#         # update edges + nodes
#         all_nodes_neighbors |= new_nodes_neighbors
#         all_edges_neighbors = [
#             (u, v, k)
#             for u, v, k in G.edges
#             if u in all_nodes_neighbors or v in all_nodes_neighbors
#         ]

#         G_u = G.subgraph(all_nodes_neighbors).copy()
        
#         return all_nodes_neighbors


def find_next_neighbors(G, edges):
    next_neighbors = set()
    for edge in edges:
        u, v, k = edge
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))

        next_neighbors.update(neighbors_u)
        next_neighbors.update(neighbors_v)

    return next_neighbors


def find_next_next_neighbors(G, edges):
    next_neighbors = find_next_neighbors(G, edges)

    next_next_neighbors = set()
    for edge in edges:
        # Find next nearest neighbors of the next nearest neighbors

        for neighbor in next_neighbors:
            next_next_neighbors.update(set(G.neighbors(neighbor)))

    return next_next_neighbors


def find_next_next_neighbors_until_connected(G, edges):

    print(f"Finding next next neighbors for edges: {edges}")

    # Step 1: collect neighbors of all given edges
    all_nodes_neighbors = set()
    for edge in edges:
        nodes_neighbors = find_next_neighbors(G, [edge])
        all_nodes_neighbors |= nodes_neighbors  # union

    # Step 2: get edges connected to those neighbors
    all_edges_neighbors = [
        (u, v, k)
        for u, v, k in G.edges
        if u in all_nodes_neighbors or v in all_nodes_neighbors
    ]

    # Step 3: induced subgraph
    G_u = G.subgraph(all_nodes_neighbors).copy()

    # Step 4: expand until connected
    while not nx.is_connected(G_u):
        new_nodes_neighbors = set()
        for e in all_edges_neighbors:
            new_nodes_neighbors |= find_next_neighbors(G, [e])

        # update edges + nodes
        all_nodes_neighbors |= new_nodes_neighbors
        all_edges_neighbors = [
            (u, v, k)
            for u, v, k in G.edges
            if u in all_nodes_neighbors or v in all_nodes_neighbors
        ]

        G_u = G.subgraph(all_nodes_neighbors).copy()

    return all_nodes_neighbors


def nodes_shortest_path_between_edges(G, edges):
    nodes_between = set()
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            u1, v1, k1 = edges[i]
            u2, v2, k2 = edges[j]
            try:
                path1 = nx.shortest_path(G, source=u1, target=u2)
                path2 = nx.shortest_path(G, source=u1, target=v2)
                path3 = nx.shortest_path(G, source=v1, target=u2)
                path4 = nx.shortest_path(G, source=v1, target=v2)
                shortest_path = min([path1, path2, path3, path4], key=len)
                nodes_between.update(shortest_path)
            except nx.NetworkXNoPath:
                continue
    return nodes_between


def find_next_next_neighbors_until_connected_limited(G, edges):

    print(f"Finding next next neighbors for edges: {edges}")

    # Step 1: collect nodes on shortest paths between all given edges
    all_nodes_neighbors = nodes_shortest_path_between_edges(G, edges)
    # Step 2: get edges connected to those neighbors
    all_edges_neighbors = [
        (u, v, k)
        for u, v, k in G.edges
        if u in all_nodes_neighbors or v in all_nodes_neighbors
    ]
    # Step 3: Add all nodes of the given edges
    for edge in edges:
        all_nodes_neighbors.add(edge[0])
        all_nodes_neighbors.add(edge[1])

    return all_nodes_neighbors
