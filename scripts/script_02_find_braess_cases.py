import scipy.sparse as sparse
from tqdm import tqdm
import csv

# --- Standard Library ---
import sys
import warnings
import pickle as pkl
import itertools as it

# --- Set repository path and autoreload ---
root_path = "../"
sys.path.append(root_path)

warnings.simplefilter(action="ignore", category=FutureWarning)


# --- Core Libraries ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import os

# --- Cartopy for map plotting ---
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy

# --- Matplotlib utilities ---
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.gridspec as gridspec

# --- PyPSA ---
import pypsa

# --- Custom Utilities ---
from utils.config import (
    prenetwork_path,
    postnetwork_path,
    results_path,
    plots_path,
    braess_path,
)


from utils.cascade_simulation import (
    solve_lpf,
)

from utils.data_handling import (
    add_flows_to_graph,
    get_matrices_from_nx_graph_multigraph,
    get_flownetwork_without_lines,
    sort_edge_indices,
    serialize_edge,
    deserialize_edge,
    serialize_edges,
    deserialize_edges,
    nx_edges_to_matrix_indices,
    construct_incidencematrix_from_orientation_multigraph,
)

SAVE_FILE = braess_path + f"braess_paradox_cases.csv"
CHECKPOINT_INTERVAL = 20  # save every 100 checked pairs


def save_cases_csv(cases, filename=SAVE_FILE):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["edge1", "edge2", "reversed_edges"])
        for e1, e2, reversed_edges in cases:
            writer.writerow(
                [
                    serialize_edge(e1),
                    serialize_edge(e2),
                    serialize_edges(reversed_edges),
                ]
            )


def load_cases_csv(filename=SAVE_FILE):
    cases = []
    if not os.path.exists(filename):
        # raise FileNotFoundError(f"File {filename} does not exist.")
        # create file
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["edge1", "edge2", "reversed_edges"])
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


def find_braess_paradox_with_reversals_and_connectivity_check(
    G, P, threshold=1e-3, filename=SAVE_FILE
):
    paradox_cases = load_cases_csv(filename=filename)
    all_edges = list(G.edges(keys=True))
    found_pairs = set(tuple(sorted((case[0], case[1]))) for case in paradox_cases)

    I_m, B_d, *_ = get_matrices_from_nx_graph_multigraph(G)
    base_flows = solve_lpf(P, B_d, I_m)
    G_base = add_flows_to_graph(G.copy(), base_flows)
    base_flow_dict = {
        (u, v, k): data["flow"] for u, v, k, data in G_base.edges(keys=True, data=True)
    }

    checked_count = 0
    with tqdm(
        total=len(all_edges) * (len(all_edges) - 1) // 2, desc="Searching edge pairs"
    ) as pbar:
        for i in range(len(all_edges)):
            for j in range(i + 1, len(all_edges)):
                pair = tuple(sorted((all_edges[i], all_edges[j])))
                if pair in found_pairs:
                    pbar.update(1)
                    continue

                e1, e2 = all_edges[i], all_edges[j]

                G_tmp = G.copy()
                G_tmp.remove_edge(*e1)
                G_tmp.remove_edge(*e2)
                if not nx.is_connected(G_tmp):
                    pbar.update(1)
                    continue

                G_12 = get_flownetwork_without_lines(G, P, [e1, e2])
                flow_dict_12 = {
                    (u, v, k): data["flow"]
                    for u, v, k, data in G_12.edges(keys=True, data=True)
                }

                G_tmp = G.copy()
                G_tmp.remove_edge(*e1)
                if not nx.is_connected(G_tmp):
                    pbar.update(1)
                    continue
                G_1 = get_flownetwork_without_lines(G, P, [e1])
                flow_dict_1 = {
                    (u, v, k): data["flow"]
                    for u, v, k, data in G_1.edges(keys=True, data=True)
                }

                G_tmp = G.copy()
                G_tmp.remove_edge(*e2)
                if not nx.is_connected(G_tmp):
                    pbar.update(1)
                    continue
                G_2 = get_flownetwork_without_lines(G, P, [e2])
                flow_dict_2 = {
                    (u, v, k): data["flow"]
                    for u, v, k, data in G_2.edges(keys=True, data=True)
                }

                # Check flow direction changes
                for edge in base_flow_dict:
                    if (
                        edge in flow_dict_1
                        and edge in flow_dict_2
                        and edge in flow_dict_12
                    ):
                        f_base = base_flow_dict[edge]
                        delta1 = flow_dict_1[edge] - f_base
                        delta2 = flow_dict_2[edge] - f_base
                        delta12 = flow_dict_12[edge] - f_base

                        if (
                            abs(delta1) < threshold
                            or abs(delta2) < threshold
                            or abs(delta12) < threshold
                        ):
                            continue  # ignore very small changes

                        # individual removals change flow in same direction, joint removal in opposite
                        if delta1 * delta2 > 0 and delta1 * delta12 < 0:
                            paradox_cases.append((e1, e2, [edge]))
                            found_pairs.add(pair)
                            save_cases_csv(paradox_cases, filename)
                            break  # only need one edge to satisfy

                pbar.update(1)
                checked_count += 1

                if checked_count % CHECKPOINT_INTERVAL == 0:
                    save_cases_csv(paradox_cases, filename)

    return paradox_cases




# Example usage:
if __name__ == "__main__":

    # parallel loops for all three sync grids
    # Great_Britain, Scandinavia, Continental_Europe

    sync_grid = "Scandinavia" # Great_Britain, Scandinavia or Continental_Europe

    print(f"Processing {sync_grid}...")
    # Load the graph and matrices for the specified sync grid
    multi_G_path = f"{postnetwork_path}/{sync_grid}_deagg_graph.pkl"

    # read pickled graph
    G = pkl.load(
        open(f"{postnetwork_path}/{sync_grid}_deagg_graph.pkl", "rb")
    )  # Your networkx.MultiGraph with attributes
    G = sort_edge_indices(G)  # Ensure edges are sorted
    G_multi_data = np.load(
        f"{postnetwork_path}/{sync_grid}_deagg_matrices.npz", allow_pickle=True
    )
    # P=P, I_m=I_m, B_d=B_d, num_parallels=num_parallels, line_limits=line_limits
    P = G_multi_data["P"]

    cases = find_braess_paradox_with_reversals_and_connectivity_check(
        G=G, P=P, filename=SAVE_FILE
    )
    print("Found paradox cases:", cases)
