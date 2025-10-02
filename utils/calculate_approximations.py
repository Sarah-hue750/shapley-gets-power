# --- Standard Library ---
import sys
import os
import warnings
import pickle as pkl

# --- Set repository path and autoreload ---
root_path = "../"
sys.path.append(root_path)

warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Core Libraries ---
import time
from tqdm import tqdm
import random

# --- Custom Utilities ---
from utils.config import (
    shap_approx_path,
)
from utils.calculate_shap import (
    get_shapley_value_all_outage_lines,
    get_shapley_value_knn_approx_all_outage_lines,
    get_shapley_taylor_all_outage_lines,
    flow_attribution_all_affected_lines_dict,
    get_shapley_taylor_knn_approx_all_outage_lines,
)


from utils.data_handling import (
    nx_edges_to_matrix_indices,
    remove_random_edges,
    get_attributes_from_flow_change,
    shuffle_knn_dict,
    shuffle_knn_dict_second_order,
    get_knn_dict,
    get_knn_dict_second_order,
)
from utils.collectivity_clustering import (
    knn_failure_subgraphs,
    knn_failure_subgraphs_second_order,
)


def approximation_comparison_multiple_outages_knn(
    G_0, P, B_d, I, num_outage_sets, num_edges=12, seed=42, file_name="mul"
):
    """

    Creates results dicts containing approximation errors, for multiple outage sets.
    Args:
        G_0 (networkx Graph): Original Graph
        P (matrix): power flows of Graph
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix
        num_outage_sets (int): number of different outage sets, for which shap approx should be calculated.
        num_edges (int): Number of outage lines per outage set. Default is 12.
        seed (int): Random seed for reproducability. Default is 42.
        file_name (string): string to be part of the filename, as well as num_edges and method. Default is "mul".
    Returns:
        results_dict (list): Contains results dict for all different outage sets.
    """
    random.seed(seed)
    results_dict = []
    seen_outages = set()  # store unique sets
    outage_sets = []
    for i in range(
        num_outage_sets
    ):  # Changed to pre-generate all outage sets to get same outage sets with same seed
        while True:
            seed1 = random.randint(0, 100000)
            outage_lines = remove_random_edges(G_0, num_edges=num_edges, seed=seed1)

            # Create a sorted tuple to ensure uniqueness
            outage_tuple = tuple(sorted(outage_lines))

            if outage_tuple not in seen_outages:
                seen_outages.add(outage_tuple)
                outage_sets.append(outage_lines)
                break  # exit the while loop if a unique set is found

    for i, outage_lines in tqdm(
        enumerate(outage_sets), total=len(outage_sets), desc="Processing outages"
    ):
        outage_lines_nx = nx_edges_to_matrix_indices(outage_lines, G_0)
        all_lines = range(0, B_d.shape[0])

        # get affected lines
        affected_lines = [line for line in all_lines if line not in outage_lines_nx]
        if num_edges > 12:
            ks = [2, 4, 6, 8, 10, 12]  # added 12
        else:
            ks = [2, 4, 6, 8]
        knn_dicts = {}
        knn_creation_times = {}
        for k in ks:
            knn_creation_start = time.time()
            knn_orig = knn_failure_subgraphs(
                graph=G_0, edge_failed=outage_lines, nearest_neighbors=k
            )
            knn_creation_time = time.time() - knn_creation_start
            knn_dicts[k] = get_knn_dict(knn_orig, G_0)
            knn_creation_times[k] = knn_creation_time

        seed2 = random.randint(0, 100000)
        results = approximation_comparison_knn(
            affected_lines=affected_lines,
            outage_lines=outage_lines_nx,
            knn_dicts=knn_dicts,
            P=P,
            B_d=B_d,
            M=set(outage_lines_nx),
            I=I,
            G=G_0,
            seed=seed2,
        )
        results_dict.append(
            {
                "outage_lines": outage_lines,
                "outage_lines_nx": outage_lines_nx,
                "results": results,
                "cluster_creation_times": knn_creation_times,
            }
        )
        file_path = os.path.join(
            shap_approx_path,
            f"{file_name}_knn_n{len(outage_lines)}_m{num_outage_sets}.pkl",
        )

        # Make sure the folder exists
        os.makedirs(shap_approx_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pkl.dump(results_dict, f)


def approximation_comparison_knn(
    affected_lines,
    outage_lines,
    knn_dicts,
    P,
    B_d,
    M,
    I,
    G,
    file_name="knn",
    seed=42,
    save_file=True,
):
    """
    Creates a result dict containing the approximation errors, shap and shap approx values for all specified affected lines and outage lines.
    Also creates for each of the approximation cluster dict a randomly permuted cluster dict with same cluster sizes as a comparison.

    Args:
        affected_lines(list): List of int indices of affected lines
        outage_lines(list): List of int indices of outage lines
        cluster_dicts (dict of dicts): Contains cluster dict as values which have line indices as keys and their cluster idx as value.
                                        Cluster dicts are indexed with some kind of resolution measure of the clustering.
        P (float matrix): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        G (networkx Graph): Corresponding Graph, necessary to get other attributes from flow change.
        file_name (string): Name with which the result dict is saved, a "_n(#number of outages).pkl" is added as well
        seed (int): Random seed for clustering. Default is 42.
        save_file (bool): If True results are saved. Default is True.
    Returns:
        results (dict): Dictionary containing approx errors, shap, shap approx etc.
    """
    results = {}
    cluster_results = []
    random.seed(seed)
    # Exact Shapley values (baseline)
    start_time = time.time()
    shap = get_shapley_value_all_outage_lines(B_d=B_d, M=M, I=I)
    shap_time = time.time() - start_time

    # Attribution of exact Shapley values
    shap_attr = flow_attribution_all_affected_lines_dict(
        affected_lines=affected_lines, P=P, B_d=B_d, I=I, value_dict=shap
    )
    results = {
        "shap_time": shap_time,
        "exact_shap_flow_change": shap_attr,
    }

    for attr in ["load", "current"]:
        shap_new = {}
        shap_rand_new = {}
        abs_rand_diffs = {}
        rel_rand_diffs = {}
        for affected_line in shap_attr:
            shap_new[affected_line] = get_attributes_from_flow_change(
                G=G, flow_change_dict=shap_attr[affected_line], attr=attr
            )
        results[f"exact_shap_{attr}"] = shap_new

    for label, knn_dict in knn_dicts.items():
        # Filter cluster_dict to outage lines
        subset_cluster = {k: v for k, v in knn_dict.items() if k in outage_lines}

        # Approximate Shapley values
        start_approx_time = time.time()
        shap_approx = get_shapley_value_knn_approx_all_outage_lines(
            B_d=B_d,
            M=M,
            I=I,
            knn_dict=knn_dict,
        )
        approx_time = time.time() - start_approx_time

        # Attribution of approximate Shapley values
        shap_approx_attr = flow_attribution_all_affected_lines_dict(
            affected_lines=affected_lines, P=P, B_d=B_d, I=I, value_dict=shap_approx
        )
        # Random equivalent
        seed = random.randint(0, 1000)
        knn_dict_rand = shuffle_knn_dict(original_knn_dict=knn_dict, seed=seed)
        start_rand_time = time.time()
        shap_rand = get_shapley_value_knn_approx_all_outage_lines(
            B_d=B_d,
            M=M,
            I=I,
            knn_dict=knn_dict_rand,
        )
        rand_time = time.time() - start_rand_time
        shap_rand_attr = flow_attribution_all_affected_lines_dict(
            affected_lines=affected_lines, P=P, B_d=B_d, I=I, value_dict=shap_rand
        )

        # Compute per-line absolute and relative differences
        all_diff = {}
        abs_diffs = {}
        rel_diffs = {}
        abs_rand_diffs = {}
        rel_rand_diffs = {}
        for key in shap_approx_attr:
            abs_diffs[key] = {
                k: shap_approx_attr[key][k]
                - shap_attr[key][k]  # Removed absolute value
                for k in shap_attr[key]
            }
            rel_diffs[key] = {
                k: abs_diffs[key][k] / abs(shap_attr[key][k])
                for k in abs_diffs[key].keys()
            }
            abs_rand_diffs[key] = {
                k: shap_rand_attr[key][k] - shap_attr[key][k]  # Removed absolute value
                for k in shap_attr[key]
            }
            rel_rand_diffs[key] = {
                k: abs_rand_diffs[key][k] / abs(shap_attr[key][k])
                for k in abs_rand_diffs[key]
            }
        all_diff[f"approx_shap_flow_change"] = shap_approx_attr
        all_diff["abs_diff_flow_change"] = abs_diffs
        all_diff["rel_diff_flow_change"] = rel_diffs
        all_diff["rand_shap_flow_change"] = shap_rand_attr
        all_diff["abs_rand_diff_flow_change"] = abs_rand_diffs
        all_diff["rel_rand_diff_flow_change"] = rel_rand_diffs
        # Different attributions
        for attr in ["load", "current"]:
            abs_diffs = {}
            rel_diffs = {}
            abs_rand_diffs = {}
            rel_rand_diffs = {}
            shap_approx_new = {}
            shap_rand_new = {}
            for key in shap_approx_attr:
                shap_approx_new[key] = get_attributes_from_flow_change(
                    G=G, flow_change_dict=shap_approx_attr[key], attr=attr
                )
                abs_diffs[key] = {
                    k: shap_approx_new[key][k]
                    - results[f"exact_shap_{attr}"][key][k]  # Removed absolute value
                    for k in results[f"exact_shap_{attr}"][key]
                }
                rel_diffs[key] = {
                    k: abs_diffs[key][k] / abs(results[f"exact_shap_{attr}"][key][k])
                    for k in abs_diffs[key].keys()
                }
                # Random equivalent
                shap_rand_new[key] = get_attributes_from_flow_change(
                    G=G, flow_change_dict=shap_rand_attr[key], attr=attr
                )
                abs_rand_diffs[key] = {
                    k: shap_rand_new[key][k]
                    - results[f"exact_shap_{attr}"][key][k]  # Removed absolute value
                    for k in results[f"exact_shap_{attr}"][key]
                }
                rel_rand_diffs[key] = {
                    k: abs_rand_diffs[key][k]
                    / abs(results[f"exact_shap_{attr}"][key][k])
                    for k in abs_rand_diffs[key]
                }
            all_diff[f"abs_diff_{attr}"] = abs_diffs
            all_diff[f"rel_diff_{attr}"] = rel_diffs
            all_diff[f"approx_shap_{attr}"] = shap_approx_new
            all_diff[f"rand_shap_{attr}"] = shap_rand_new
            all_diff[f"abs_rand_diff_{attr}"] = abs_rand_diffs
            all_diff[f"rel_rand_diff_{attr}"] = rel_rand_diffs

        # Store all results for this approximation method
        cluster_results.append(
            {
                "label": label,
                "approx_time": approx_time,
                "differences": all_diff,
                "knn_dict": knn_dict,
                "rand_time": rand_time,
                "knn_dict_rand": knn_dict_rand,
            }
        )
    results["cluster_results"] = cluster_results
    # Save
    if save_file:
        file_path = os.path.join(
            shap_approx_path, f"{file_name}_n{len(outage_lines)}.pkl"
        )

        # Make sure the folder exists
        os.makedirs(shap_approx_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pkl.dump(results, f)
    return results


def approximation_comparison_taylor_knn_multiple_outages(
    G_0, P, B_d, I, num_outage_sets, num_edges=12, seed=42, file_name="mul"
):
    random.seed(seed)
    results_dict = []
    seen_outages = set()  # store unique sets
    outage_sets = []
    for i in range(
        num_outage_sets
    ):  # Changed to pre-generate all outage sets to get same outage sets with same seed
        while True:
            seed1 = random.randint(0, 100000)
            outage_lines = remove_random_edges(G_0, num_edges=num_edges, seed=seed1)

            # Create a sorted tuple to ensure uniqueness
            outage_tuple = tuple(sorted(outage_lines))

            if outage_tuple not in seen_outages:
                seen_outages.add(outage_tuple)
                outage_sets.append(outage_lines)
                break  # exit the while loop if a unique set is found

    for i, outage_lines in tqdm(
        enumerate(outage_sets), total=len(outage_sets), desc="Processing outages"
    ):
        outage_lines_nx = nx_edges_to_matrix_indices(outage_lines, G_0)
        all_lines = range(0, B_d.shape[0])

        # get affected lines
        affected_lines = [line for line in all_lines if line not in outage_lines_nx]
        if num_edges > 12:
            ks = [2, 4, 6, 8, 10, 12]  # added 12
        else:
            ks = [2, 4, 6, 8]
        knn_dicts = {}
        knn_creation_times = {}
        for k in ks:
            knn_creation_start = time.time()
            knn_orig = knn_failure_subgraphs_second_order(
                graph=G_0, edge_failed=outage_lines, nearest_neighbors=k
            )
            knn_creation_time = time.time() - knn_creation_start
            knn_dicts[k] = get_knn_dict_second_order(knn_orig, G_0)
            knn_creation_times[k] = knn_creation_time

        seed2 = random.randint(0, 100000)
        results = approximation_comparison_taylor_knn(
            affected_lines=affected_lines,
            outage_lines=outage_lines_nx,
            knn_dicts=knn_dicts,
            P=P,
            B_d=B_d,
            M=set(outage_lines_nx),
            I=I,
            G=G_0,
            seed=seed2,
        )
        results_dict.append(
            {
                "outage_lines": outage_lines,
                "outage_lines_nx": outage_lines_nx,
                "results": results,
                "cluster_creation_times": knn_creation_times,
            }
        )
        file_path = os.path.join(
            shap_approx_path,
            f"{file_name}_knn_taylor_n{len(outage_lines)}_m{num_outage_sets}.pkl",
        )

        # Make sure the folder exists
        os.makedirs(shap_approx_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pkl.dump(results_dict, f)


def approximation_comparison_taylor_knn(
    affected_lines,
    outage_lines,
    knn_dicts,
    P,
    B_d,
    M,
    I,
    G,
    file_name="test",
    seed=42,
    save_file=True,
):
    """
    Creates a result dict containing the approximation errors, shap and shap approx taylor values for all specified affected lines and outage lines.
    Also creates for each of the approximation knn dict a randomly permuted knn dict with same cluster sizes as a comparison.

    Args:
        affected_lines(list): List of int indices of affected lines
        outage_lines(list): List of int indices of outage lines
        knn_dicts (dict of dicts): Contains knn dict as values which have line indices as keys and nearest neighbors as values.
    Returns:
        results (dict): Dictionary containing approx errors, shap taylor , shap taylor approx
    """
    results = {}
    cluster_results = []
    random.seed(seed)
    # Exact Shapley taylor (baseline)
    start_time = time.time()
    shap = get_shapley_taylor_all_outage_lines(B_d=B_d, M=M, I=I)
    shap_time = time.time() - start_time

    # Attribution of exact Shapley taylor
    shap_attr = flow_attribution_all_affected_lines_dict(
        affected_lines=affected_lines, P=P, B_d=B_d, I=I, value_dict=shap
    )
    results = {
        "shap_time": shap_time,
        "exact_shap_flow_change": shap_attr,
    }

    for attr in ["load", "current"]:
        shap_new = {}
        shap_rand_new = {}
        abs_rand_diffs = {}
        rel_rand_diffs = {}
        for affected_line in shap_attr:
            shap_new[affected_line] = get_attributes_from_flow_change(
                G=G, flow_change_dict=shap_attr[affected_line], attr=attr
            )
        results[f"exact_shap_{attr}"] = shap_new

    for label, knn_dict in knn_dicts.items():
        # Filter cluster_dict to outage lines
        subset_cluster = {k: v for k, v in knn_dict.items() if k in outage_lines}

        # Approximate Shapley taylor
        start_approx_time = time.time()
        shap_approx = get_shapley_taylor_knn_approx_all_outage_lines(
            B_d=B_d, M=M, I=I, knn_dict=knn_dict
        )
        approx_time = time.time() - start_approx_time

        # Attribution of approximate Shapley taylor
        shap_approx_attr = flow_attribution_all_affected_lines_dict(
            affected_lines=affected_lines, P=P, B_d=B_d, I=I, value_dict=shap_approx
        )
        # Random equivalent
        seed = random.randint(0, 1000)
        knn_dict_rand = shuffle_knn_dict_second_order(
            original_knn_dict=knn_dict, seed=seed
        )
        start_rand_time = time.time()
        shap_rand = get_shapley_taylor_knn_approx_all_outage_lines(
            B_d=B_d, M=M, I=I, knn_dict=knn_dict_rand
        )
        rand_time = time.time() - start_rand_time
        shap_rand_attr = flow_attribution_all_affected_lines_dict(
            affected_lines=affected_lines, P=P, B_d=B_d, I=I, value_dict=shap_rand
        )

        # Compute per-line absolute and relative differences
        all_diff = {}
        abs_diffs = {}
        rel_diffs = {}
        abs_rand_diffs = {}
        rel_rand_diffs = {}
        for key in shap_approx_attr:
            abs_diffs[key] = {
                k: shap_approx_attr[key][k]
                - shap_attr[key][k]  # Removed absolute value
                for k in shap_attr[key]
            }
            rel_diffs[key] = {
                k: abs_diffs[key][k] / abs(shap_attr[key][k])
                for k in abs_diffs[key].keys()
            }
            abs_rand_diffs[key] = {
                k: shap_rand_attr[key][k] - shap_attr[key][k]  # Removed absolute value
                for k in shap_attr[key]
            }
            rel_rand_diffs[key] = {
                k: abs_rand_diffs[key][k] / abs(shap_attr[key][k])
                for k in abs_rand_diffs[key]
            }
        all_diff[f"approx_shap_flow_change"] = shap_approx_attr
        all_diff["abs_diff_flow_change"] = abs_diffs
        all_diff["rel_diff_flow_change"] = rel_diffs
        all_diff["rand_shap_flow_change"] = shap_rand_attr
        all_diff["abs_rand_diff_flow_change"] = abs_rand_diffs
        all_diff["rel_rand_diff_flow_change"] = rel_rand_diffs
        # Different attributions
        for attr in ["load", "current"]:
            abs_diffs = {}
            rel_diffs = {}
            abs_rand_diffs = {}
            rel_rand_diffs = {}
            shap_approx_new = {}
            shap_rand_new = {}
            for key in shap_approx_attr:
                shap_approx_new[key] = get_attributes_from_flow_change(
                    G=G, flow_change_dict=shap_approx_attr[key], attr=attr
                )
                abs_diffs[key] = {
                    k: shap_approx_new[key][k]
                    - results[f"exact_shap_{attr}"][key][k]  # Removed absolute value
                    for k in results[f"exact_shap_{attr}"][key]
                }
                rel_diffs[key] = {
                    k: abs_diffs[key][k] / abs(results[f"exact_shap_{attr}"][key][k])
                    for k in abs_diffs[key].keys()
                }
                # Random equivalent
                shap_rand_new[key] = get_attributes_from_flow_change(
                    G=G, flow_change_dict=shap_rand_attr[key], attr=attr
                )
                abs_rand_diffs[key] = {
                    k: shap_rand_new[key][k]
                    - results[f"exact_shap_{attr}"][key][k]  # Removed absolute value
                    for k in results[f"exact_shap_{attr}"][key]
                }
                rel_rand_diffs[key] = {
                    k: abs_rand_diffs[key][k]
                    / abs(results[f"exact_shap_{attr}"][key][k])
                    for k in abs_rand_diffs[key]
                }
            all_diff[f"abs_diff_{attr}"] = abs_diffs
            all_diff[f"rel_diff_{attr}"] = rel_diffs
            all_diff[f"approx_shap_{attr}"] = shap_approx_new
            all_diff[f"rand_shap_{attr}"] = shap_rand_new
            all_diff[f"abs_rand_diff_{attr}"] = abs_rand_diffs
            all_diff[f"rel_rand_diff_{attr}"] = rel_rand_diffs

        # Store all results for this approximation method
        cluster_results.append(
            {
                "label": label,
                "approx_time": approx_time,
                "differences": all_diff,
                "cluster_dict": knn_dict,
                "rand_time": rand_time,
                "cluster_dict_rand": knn_dict_rand,
            }
        )
    results["cluster_results"] = cluster_results
    # Save
    if save_file:
        file_path = os.path.join(
            shap_approx_path, f"{file_name}_knn_taylor_n{len(outage_lines)}.pkl"
        )

        # Make sure the folder exists
        os.makedirs(shap_approx_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pkl.dump(results, f)
    return results
