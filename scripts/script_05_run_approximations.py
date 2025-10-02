# --- Standard Library ---
import sys
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import warnings
import pickle as pkl

# --- Set repository path and autoreload ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)
from multiprocessing import freeze_support

warnings.simplefilter(action="ignore", category=FutureWarning)


# --- Core Libraries ---
import numpy as np


# --- Custom Utilities ---
from utils.config import (
    prenetwork_path,
    postnetwork_path,
    results_path,
    shap_approx_path,
    plots_path,
)
from utils.cascade_simulation import (
    solve_lpf,
)
from utils.data_handling import (
    add_flows_to_graph,
    remove_random_edges,
    nx_edges_to_matrix_indices,
    get_knn_dict,
)
from utils.collectivity_clustering import (
    knn_failure_subgraphs
)

from utils.calculate_approximations import (
    approximation_comparison_multiple_outages_knn,
    approximation_comparison_knn,
)


# Import Grid
sync_grid = "Scandinavia"  # Great_Britain, Scandinavia or Continental_Europe

multi_G_path = f"{postnetwork_path}/{sync_grid}_deagg_graph.pkl"

# read pickled graph
G_multi = pkl.load(open(f"{postnetwork_path}/{sync_grid}_deagg_graph.pkl", "rb"))
G_multi_data = np.load(
    f"{postnetwork_path}/{sync_grid}_deagg_matrices.npz", allow_pickle=True
)
P = G_multi_data["P"]
I_m = G_multi_data["I_m"].item()
B_d = G_multi_data["B_d"].item()
num_parallels = G_multi_data["num_parallels"]
line_limits = G_multi_data["line_limits"]


# calculate undisrupted network
flows_0 = solve_lpf(P=P, B_d=B_d, I=I_m)
G_0 = add_flows_to_graph(G_multi, flows_0)


# Create single outage set approximations
num_edges = 16
rand_edges = remove_random_edges(G_0, num_edges=num_edges, seed=50)
random_outage_lines = nx_edges_to_matrix_indices(rand_edges, G_0)

all_lines = range(0, B_d.shape[0])
affected_lines = [line for line in all_lines if line not in random_outage_lines]
ks = [2, 4, 6, 8, 10, 12]
knn_dicts = {}
for k in ks:
    knn_orig = knn_failure_subgraphs(
        graph=G_0, edge_failed=rand_edges, nearest_neighbors=k
    )
    knn_dicts[k] = get_knn_dict(knn_orig, G_0)



# Create multiple outage sets approximations
seed = 50
if __name__ == "__main__":
    freeze_support()
    approximation_comparison_multiple_outages_knn(
        G_0=G_0,
        P=P,
        B_d=B_d,
        I=I_m,
        num_outage_sets=100,
        num_edges=16,
        file_name="s50_final",
        seed=seed,
    )
    approximation_comparison_knn(
    affected_lines=affected_lines,
    outage_lines=random_outage_lines,
    knn_dicts=knn_dicts,
    P=P,
    B_d=B_d,
    M=set(random_outage_lines),
    I=I_m,
    G=G_0,
    file_name="s50_final",
    seed = seed,
    )
    
    
