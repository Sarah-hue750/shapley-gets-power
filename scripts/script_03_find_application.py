# --- Set repository path and autoreload ---
import sys
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
# --- Set repository path and autoreload ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

from utils.config import (
    prenetwork_path,
    postnetwork_path,
    results_path,
    plots_path,
    application_path,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random
import pickle as pkl
import numpy as np
import csv
import networkx as nx
import itertools
from tqdm import tqdm

from utils.cascade_simulation import (
    solve_lpf,
)
from utils.data_handling import (
    add_flows_to_graph,
    get_matrices_from_nx_graph_multigraph,
    sort_edge_indices,
    serialize_edge,
    deserialize_edge,
    serialize_edges,
    deserialize_edges,
    nx_edges_to_matrix_indices,
    construct_incidencematrix_from_orientation_multigraph,
    get_flownetwork_without_lines,
)


SAVE_FILE = application_path + f"application_cases.csv"
CHECKPOINT_INTERVAL = 20  # save every 100 checked pairs
    
def save_quadruple_cases_csv(cases, filename=SAVE_FILE):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["edge1", "edge2", "edge3", "edge4", "reversed_edges", "non_outage_line"]
        )
        for e1, e2, e3, e4, reversed_edges, non_outage_line in cases:
            writer.writerow(
                [
                    serialize_edge(e1),
                    serialize_edge(e2),
                    serialize_edge(e3),
                    serialize_edge(e4),
                    serialize_edges(reversed_edges),
                    serialize_edges(non_outage_line),
                ]
            )


def load_quadruple_cases_csv(filename=SAVE_FILE):
    cases = []
    if not os.path.exists(filename):
        # raise FileNotFoundError(f"File {filename} does not exist.")
        # create file
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "edge1",
                    "edge2",
                    "edge3",
                    "edge4",
                    "reversed_edges",
                    "non_outage_line",
                ]
            )
    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                e1 = deserialize_edge(row["edge1"])
                e2 = deserialize_edge(row["edge2"])
                e3 = deserialize_edge(row["edge3"])
                e4 = deserialize_edge(row["edge4"])
                reversed_edges = deserialize_edges(row["reversed_edges"])
                non_outage_line = deserialize_edges(row["non_outage_line"])
                cases.append((e1, e2, e3, e4, reversed_edges, non_outage_line))
    except FileNotFoundError:
        pass
    return cases


def check_quadrupel(quadruple, found_quadruples, G, P, base_flow_dict, threshold):
    """
    Check a single quadruple for the paradox of choice.
    Returns the paradox case if found, otherwise None.
    Args:
        quadruple (tuple): tuple of four edges to check
        found_quadruples (set): set of already found quadruples to avoid duplicates
        G (networkx.MultiGraph): power grid graph
        P (list): power injection vector
        base_flow_dict (dict): base flow dictionary
        threshold (float): threshold for significant influence
    Returns:
        tuple or None: paradox case if found, otherwise None"""

    if quadruple in found_quadruples:
        return None

    e1, e2, e3, e4 = quadruple
    quadruple_edges = [e1, e2, e3, e4]

    G_tmp = G.copy()
    G_tmp.remove_edge(*e1)
    G_tmp.remove_edge(*e2)
    G_tmp.remove_edge(*e3)
    G_tmp.remove_edge(*e4)
    if not nx.is_connected(G_tmp):
        return None

    disconnected = False
    dict_single_failure = {}
    for i, e in enumerate([e1, e2, e3, e4]):
        G_tmp = G.copy()
        G_tmp.remove_edge(*e)
        if not nx.is_connected(G_tmp):
            disconnected = True
            break
        G_i = get_flownetwork_without_lines(G, P, [e])
        flow_dict_i = {
            (u, v, k): data["flow"] for u, v, k, data in G_i.edges(keys=True, data=True)
        }
        quadruple_not_i = [edge for edge in quadruple if edge != e]
        G_not_i = get_flownetwork_without_lines(G, P, quadruple_not_i)
        flow_dict_not_i = {
            (u, v, k): data["flow"]
            for u, v, k, data in G_not_i.edges(keys=True, data=True)
        }

        dict_single_failure[i] = {
            "edge": e,
            "G_i": G_i,
            "flow_dict_i": flow_dict_i,
            "G_not_i": G_not_i,
            "flow_dict_not_i": flow_dict_not_i,
        }

    if disconnected:
        return None

    G_1234 = get_flownetwork_without_lines(G, P, [e1, e2, e3, e4])
    flow_dict_1234 = {
        (u, v, k): data["flow"] for u, v, k, data in G_1234.edges(keys=True, data=True)
    }

    monitored_edges = None
    # Check flow direction changes
    for edge in base_flow_dict:
        if edge in flow_dict_1234:
            if abs(flow_dict_1234[edge]) <= G_1234.edges[edge]["s_nom"]:
                continue
            elif monitored_edges is None:
                monitored_edges = [edge]
                continue
            else:
                monitored_edges.append(edge)
                continue
    if monitored_edges is None:
        return None
    if len(monitored_edges) > 1:
        return None  # only consider single overloads for now
    for monitored_edge in monitored_edges:
        f_base = base_flow_dict[monitored_edge]
        overload_sign = np.sign(flow_dict_1234[monitored_edge])
        no_candidate = False
        deltas = {
            i: single_failure["flow_dict_i"][monitored_edge] - f_base
            for i, single_failure in dict_single_failure.items()
        }
        deltas_scaled = {i: v * overload_sign for i, v in deltas.items()}
        deltas_not_i = {
            i: single_failure["flow_dict_not_i"][monitored_edge] - f_base
            for i, single_failure in dict_single_failure.items()
        }
        deltas_not_i_scaled = {i: v * overload_sign for i, v in deltas_not_i.items()}
        for i, single_failure in dict_single_failure.items():
            if abs(deltas[i]) < threshold:  # Check significant influence of lines
                no_candidate = True
                continue
        # Extract line that best reduces overload
        argmin = min(deltas_not_i_scaled, key=deltas_not_i_scaled.get)
        if not no_candidate:
            single_failure = dict_single_failure[argmin]
            # Check that overloading is prevented by adding edge i
            if (
                abs(single_failure["flow_dict_not_i"][monitored_edge])
                > single_failure["G_not_i"].edges[monitored_edge]["s_nom"]
            ):
                continue

            # Check if direct effect is the strongest and in the right direction
            is_max = all(
                deltas_scaled[argmin] >= v
                for j, v in deltas_scaled.items()
                if j != argmin
            )

            if not is_max:
                case = (e1, e2, e3, e4, [monitored_edge], [single_failure["edge"]])
                return case, quadruple


def find_application_example_parallel(
    G, P, threshold=1e-3, filename=SAVE_FILE, seed=50
):
    """Find application examples of the paradox of choice in a power grid.
    Args:
        G (networkx.MultiGraph): power grid graph
        P (list): power injection vector
        threshold (float): threshold for significant influence
        filename (str): filename to save/load paradox cases
        seed (int): random seed for reproducibility
    Returns:
        list: list of paradox cases found
    """
    

    paradox_cases = load_quadruple_cases_csv(filename=filename)
    all_edges = list(G.edges(keys=True))
    found_quadruples = set(
        tuple(sorted((case[0], case[1], case[2], case[3]))) for case in paradox_cases
    )
    I_m, B_d, *_ = get_matrices_from_nx_graph_multigraph(G)
    base_flows = solve_lpf(P, B_d, I_m)
    G_base = add_flows_to_graph(G.copy(), base_flows)
    base_flow_dict = {
        (u, v, k): data["flow"] for u, v, k, data in G_base.edges(keys=True, data=True)
    }

    # quadruples = [tuple(sorted(q)) for q in itertools.combinations(all_edges, 4)] #to large for scandinavia
    num_samples = 100_000  # how many quadruples you want to sample
    quadruples = set()
    random.seed(seed)
    while len(quadruples) < num_samples:
        q = tuple(sorted(random.sample(all_edges, 4)))
        quadruples.add(q)

    # for quadruple in tqdm(quadruples):

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                check_quadrupel, q, found_quadruples, G, P, base_flow_dict, threshold
            ): q
            for q in quadruples
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()  # no try/except needed unless real errors expected
            if result is None:
                continue  # skip if check_quadrupel intentionally returned None

            case, quadruple = result
            paradox_cases.append(case)
            found_quadruples.add(quadruple)
            save_quadruple_cases_csv(paradox_cases, filename)
        return paradox_cases



# Example usage:
if __name__ == "__main__":

    sync_grid = "Scandinavia"
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

    cases = find_application_example_parallel(
        G=G, P=P, filename=SAVE_FILE, seed=46
    )
    print("Found paradox cases:", cases)
