# --- Standard Library ---
import sys
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import warnings
import pickle as pkl

# --- Set repository path and autoreload ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

warnings.simplefilter(action="ignore", category=FutureWarning)


# --- Core Libraries ---
import numpy as np

# --- Matplotlib utilities ---
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt


# --- Custom Utilities ---
from utils.config import (
    prenetwork_path,
    postnetwork_path,
    results_path,
    shap_approx_path,
    plots_path,
)

from utils.data_handling import (
    add_flows_to_graph,
    nx_edges_to_matrix_indices,
    remove_random_edges,
)
from utils.cascade_simulation import (
    solve_lpf,
)
from utils.plotting import (
    approximation_figure_compare_random,
    plot_mult_error,
)

# Import the Scandinavian grid
sync_grid = "Scandinavia"  # Great_Britain, Scandinavia or Continental_Europe

multi_G_path = f"{postnetwork_path}/{sync_grid}_deagg_graph.pkl"

# read pickled graph
G_multi = pkl.load(open(f"{postnetwork_path}/{sync_grid}_deagg_graph.pkl", "rb"))
G_multi_data = np.load(
    f"{postnetwork_path}/{sync_grid}_deagg_matrices.npz", allow_pickle=True
)
# P=P, I_m=I_m, B_d=B_d, num_parallels=num_parallels, line_limits=line_limits
P = G_multi_data["P"]
I_m = G_multi_data["I_m"].item()
B_d = G_multi_data["B_d"].item()
num_parallels = G_multi_data["num_parallels"]
line_limits = G_multi_data["line_limits"]


# calculate undisrupted network
flows_0 = solve_lpf(P=P, B_d=B_d, I=I_m)
G_0 = add_flows_to_graph(G_multi, flows_0)


# Create single outage line set plot
print("Creating single outage line set plot...")
file_path_single = "s50_final_n16.pkl"
figure_path_single = "s50_final_n16.pdf"
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Dark2.colors)
with open(os.path.join(shap_approx_path, file_path_single), "rb") as f:
    results = pkl.load(f)

figure_path = os.path.join(plots_path, figure_path_single)
title = r"$\lambda_{e,l}$"

num_edges = 16
rand_edges = remove_random_edges(G_0, num_edges=num_edges, seed=50)
random_outage_lines = nx_edges_to_matrix_indices(rand_edges, G_0)

all_lines = range(0, B_d.shape[0])
affected_lines = [line for line in all_lines if line not in random_outage_lines]

approximation_figure_compare_random(
    results,
    outage_lines=rand_edges,
    G_0=G_0,
    chosen_cluster_label=10,
    file_path=figure_path,
    title=title,
)
for cluster_result in results["cluster_results"]:
    print("Label: ", cluster_result["label"])
    print("Time: ", cluster_result["approx_time"])
    print("Random Time: ", cluster_result["rand_time"])
print("Shap Time: ", results["shap_time"])

# Create multiple outage line sets plot
file_path_mult = "s50_final_knn_n16_m100.pkl"
figure_path_mult = "s50_final_knn_n16_m100.pdf"

with open(
    os.path.join(shap_approx_path, file_path_mult),
    "rb",
) as f:
    results_dict = pkl.load(f)


figure_path = os.path.join(plots_path, figure_path_mult)
plot_mult_error(results_dict=results_dict, threshold=7, file_path=figure_path)
