import itertools as it
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import pickle as pkl


# Helper functions
def get_L_inv(B_d, I):
    """
    Calculate Inverse Laplacian.
    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix
    Returns
        L_inv (matrix): Inverse Laplacian matrix.
    """
    L = I.dot(B_d).dot(I.T)
    L = L.toarray()
    L_inv = np.linalg.pinv(L)  # Use pseudo-inverse for numerical stability
    return L_inv


def weight_factor(subset_length, M):
    """
    Calculate the weight factor for a given subset length and set of failed lines.

    Args:
        subset_length (int): The length of the subset.
        M (set): Set of failed lines.

    Returns:
        float: The weight factor.
    """
    return (
        math.factorial(subset_length)
        * math.factorial(len(M) - subset_length - 1)
        / math.factorial(len(M))
    )


def get_L_subset_inv(subset, L_inv, B_d, I):
    """
    Get the inverse of the Laplacian matrix for a subset of lines which experience outage.

    Args:
        subset (tuple): A tuple of indices representing the subset of lines.
        L_inv (matrix): Laplacian matrix.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix.
    Returns:
        L_subset_inv (matrix): The inverse of the Laplacian matrix for the subset.
    """
    vector_subset_list = []
    delta_b_subset_list = []
    for l in subset:

        delta_b_l = -B_d[l, l]
        delta_b_subset_list.append(delta_b_l)
        vector_l = I.dot(np.eye(B_d.shape[0])[l])
        vector_subset_list.append(vector_l)

    # Calculate Woodbury matrix identity
    if len(vector_subset_list) == 0:
        L_subset_inv = L_inv
    else:
        U = np.array(vector_subset_list).T
        A = np.diag(delta_b_subset_list)
        L_subset_inv = L_inv - L_inv.dot(U).dot(
            np.linalg.inv(np.linalg.inv(A) + U.T.dot(L_inv).dot(U))
        ).dot(U.T).dot(L_inv)
    return L_subset_inv


# Flow attributions
def flow_attribution(affected_line, P, B_d, I, value):
    """
    Performs flow attribution of a Shapley value/ Shapley taylor or direct effect Matrix onto an effected line.

    Args:
        affected_line (int): The affected line number to get the attribution for.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix
        value (matrix): Matrix that should be projected on attributed line (Shapley value, Shapley taylor or direct effect)

    Returns:
        flow_attribution_value (float): flow attribution of the affected line for the given value matrix
    """
    b_affected_line = B_d[affected_line, affected_line]
    vector_affected_line = I.dot(np.eye(B_d.shape[0])[affected_line])
    flow_attribution_value = b_affected_line * vector_affected_line.T.dot(value).dot(P)
    return flow_attribution_value


def flow_attribution_dict(affected_line, P, B_d, I, value_dict):
    """
    Performs flow attribution of a Shapley value/ Shapley taylor or direct effect dict of matrices onto an effected line.

    Args:
        affected_line (int): The affected line number to get the attribution for.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix
        value_dict (dict): Dictonary of matrices that should be projected on attributed line (Shapley value, Shapley taylor or direct effect)

    Returns:
        flow_attribution_value (float): flow attribution of the affected line for the given value matrix
    """
    flow_attribution_values = {}
    for key in value_dict.keys():
        flow_attribution_values[key] = flow_attribution(
            affected_line=affected_line, P=P, B_d=B_d, I=I, value=value_dict[key]
        )
    return flow_attribution_values


def flow_attribution_all_affected_lines(affected_lines, P, B_d, I, value):
    """
    Performs flow attribution of a Shapley value/ Shapley taylor or direct effect Matrix onto all effected lines.

    Args:
        affected_lines (int list): list of affected lines.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix
        value (matrix): Matrix that should be projected on attributed line (Shapley value, Shapley taylor or direct effect)

    Returns:
        flow_attribution_values (dict): dict with affected line as key and flow attribution value as value
    """
    flow_attribution_values = {}
    for affected_line in affected_lines:
        flow_attribution_values[affected_line] = flow_attribution(
            affected_line=affected_line, P=P, B_d=B_d, I=I, value=value
        )
    return flow_attribution_values


def flow_attribution_all_affected_lines_dict(affected_lines, P, B_d, I, value_dict):
    """
    Performs flow attribution of a Shapley value/ Shapley taylor or direct effect Matrix onto all effected lines.

    Args:
        affected_lines (int list): list of affected lines.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix
        value_dict (dict): dict with affected line as key and value as value

    Returns:
        flow_attribution_values (dict): dict with affected line as key and flow attribution value as value
    """
    flow_attribution_values = {}
    for key in value_dict.keys():
        flow_attribution_values_key = {}
        for affected_line in affected_lines:
            flow_attribution_values_key[affected_line] = flow_attribution(
                affected_line=affected_line, P=P, B_d=B_d, I=I, value=value_dict[key]
            )
        flow_attribution_values[key] = flow_attribution_values_key
    return flow_attribution_values


# Real power flows
def get_real_power_flow_change(affected_line, P, B_d, M, I, L_inv=None):
    """
    Calculates the real power flow change of an affected line when the lines in M experience outage.

    Args:
        affected_line (int): The affected line number to get the attribution for.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        real_power_flow_change (float): Change of real power flow of an affected line due to outages of lines M
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    L_M_inv = get_L_subset_inv(subset=M, L_inv=L_inv, B_d=B_d, I=I)
    b_affected_line = B_d[affected_line, affected_line]
    vector_affected_line = I.dot(np.eye(B_d.shape[0])[affected_line])
    real_power_flow_change = b_affected_line * vector_affected_line.T.dot(
        L_M_inv - L_inv
    ).dot(P)
    return real_power_flow_change


# Direct effect
def get_direct_effect(outage_line, B_d, I, L_inv=None):
    """
    Calculates direct effect of line outage.

    Args:
        outage_line (int): The outage_line number to get the direct effect for.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        Direct effect of line outage.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    L_M_inv = get_L_subset_inv(subset={outage_line}, L_inv=L_inv, B_d=B_d, I=I)
    direct_effect = L_M_inv - L_inv
    return direct_effect


def get_flow_attribution_direct_effect(
    affected_line, outage_line, P, B_d, I, L_inv=None
):
    """
    Calculates direct effect of line outage on to affected line.

    Args:
        affected_line (int): The affected line number to get the attribution for.
        outage_line (int): The outage_line number to get the direct effect for.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        Direct effect of line outage on affected line.
    """
    direct_effect = get_direct_effect(
        outage_line=outage_line, B_d=B_d, I=I, L_inv=L_inv
    )
    flow_attribution_direct_effect = flow_attribution(
        affected_line=affected_line, P=P, B_d=B_d, I=I, value=direct_effect
    )
    return flow_attribution_direct_effect


def get_flow_attribution_direct_effect_all_affected_lines(
    affected_lines, outage_line, P, B_d, I, L_inv=None
):
    """
    Calculates direct effect of line outage for all lines in a list of affected lines.

    Args:
        affected_lines (int list): list of affected lines.
        outage_line (int): The outage_line number to get the direct effect for.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        Direct effect for all lines in a list of affected lines.
    """
    direct_effects = {}
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    direct_effect = get_direct_effect(
        outage_line=outage_line, B_d=B_d, I=I, L_inv=L_inv
    )
    for affected_line in affected_lines:
        direct_effects[affected_line] = flow_attribution(
            affected_line=affected_line, P=P, B_d=B_d, I=I, value=direct_effect
        )
    return direct_effects


# Shap value
def get_shapley_value(outage_line, B_d, M, I, L_inv=None):
    """
    Get the attribution for a specific line in a list of lines.

    Args:
        outage_line (int): The outage_line number to get the attribution for.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        float: The attribution for the specified outage_line.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    # Iterate over all subset lengths
    if outage_line in M:
        M_new = M - {outage_line}
    else:
        raise ValueError(f"Element '{outage_line}' not found in set M")
    phi_subset_list = []
    for subset_length in range(len(M_new) + 1):
        for subset in it.combinations(M_new, subset_length):
            W = weight_factor(subset_length, M)
            if subset_length == 0:
                L_subset_inv = L_inv
            else:
                L_subset_inv = get_L_subset_inv(subset, L_inv, B_d, I)

            # Calculate for subset with outage_line removed
            L_subset_outage_line_inv = get_L_subset_inv(
                subset + (outage_line,), L_inv, B_d, I
            )
            # Calculate phi for this subset
            phi_subset = W * (L_subset_outage_line_inv - L_subset_inv)
            phi_subset_list.append(phi_subset)
    phi = sum(phi_subset_list)

    return phi


def get_shapley_value_all_outage_lines(B_d, M, I, L_inv=None):
    """
    Calculates shapley value for all outage lines using multiprocessing.
    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        dict: The shapley value for all outages with outage lines as keys
    """
    shapley_values = {}
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    # Partially apply arguments other than outage_line
    partial_func = partial(get_shapley_value, B_d=B_d, M=M, I=I, L_inv=L_inv)

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(partial_func, outage_line): outage_line for outage_line in M
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            outage_line = futures[future]
            shapley_values[outage_line] = future.result()

    return shapley_values


def get_flow_attribution_shapley_value(
    affected_line, outage_line, P, B_d, M, I, L_inv=None
):
    """
    Get the attribution for a specific flow on a line in a list of lines.

    Args:
        affected_line (int): The affected line number to get the attribution for.
        outage_line (int): The outage_line number to get the attribution for.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        float: The attribution for the specified flow on the affected line.
    """
    phi = get_shapley_value(outage_line=outage_line, B_d=B_d, M=M, I=I, L_inv=L_inv)
    phi_affected_line = flow_attribution(
        affected_line=affected_line, P=P, B_d=B_d, I=I, value=phi
    )
    return phi_affected_line


def get_flow_attribution_shapley_value_all_outage_lines(
    affected_line, P, B_d, M, I, L_inv=None
):
    """
    Get the Shapley value for all lines in a list of failed lines.

    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        dict: A dictionary with line indices as keys and their Shapley values as values.
    """
    shapley_values = {}
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    for outage_line in tqdm(M, total=len(M)):
        # print(f"Calculating Shapley value for outage line {outage_line}")
        shapley_values[outage_line] = get_flow_attribution_shapley_value(
            affected_line=affected_line,
            outage_line=outage_line,
            P=P,
            B_d=B_d,
            M=M,
            I=I,
            L_inv=L_inv,
        )
    return shapley_values


def get_flow_attribution_shapley_value_all_affected_lines(
    affected_lines, outage_line, P, B_d, M, I, L_inv=None
):
    """
    Get the Shapley value for all lines in a list of affected lines.

    Args:
        affected_lines (int list): list of affected lines.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        plot (boolean): If True a plot of shapley values is shown. Default is True.
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        dict: A dictionary with line indices as keys and their Shapley values as values.
    """
    shapley_values = {}
    phi = get_shapley_value(outage_line=outage_line, B_d=B_d, M=M, I=I, L_inv=L_inv)
    for affected_line in affected_lines:
        shapley_values[affected_line] = flow_attribution(
            affected_line=affected_line, P=P, B_d=B_d, I=I, value=phi
        )
    return shapley_values


# Shap Taylor


def get_shapley_taylor(outage_line_1, outage_line_2, B_d, M, I, L_inv=None):
    """
    Get the Shapley-taylor attribution for two outage lines.

    Args:
        outage_line_1 (int): The first outage line number.
        outage_line_2 (int): The second outage line number.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        float: The Shapley-taylor attribution for the two outage lines.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    # Iterate over all subset lengths
    if outage_line_1 in M and outage_line_2 in M:
        M_new = M - {outage_line_1, outage_line_2}
    else:
        raise ValueError(f"Element '{outage_line_1, outage_line_2}' not found in set M")
    phi_taylor_subset_list = []
    for subset_length in range(len(M_new) + 1):
        for subset in it.combinations(M_new, subset_length):
            W = weight_factor(subset_length, M)
            if subset_length == 0:
                L_subset_inv = L_inv
            else:
                L_subset_inv = get_L_subset_inv(subset, L_inv, B_d, I)
            # Calculate for subset with outage_line added
            L_subset_outage_line_1_inv = get_L_subset_inv(
                subset + (outage_line_1,), L_inv, B_d, I
            )
            L_subset_outage_line_2_inv = get_L_subset_inv(
                subset + (outage_line_2,), L_inv, B_d, I
            )
            L_subset_outage_lines_inv = get_L_subset_inv(
                subset + (outage_line_1, outage_line_2), L_inv, B_d, I
            )
            phi_taylor_subset = (
                2
                * W
                * (
                    L_subset_outage_lines_inv
                    + L_subset_inv
                    - L_subset_outage_line_1_inv
                    - L_subset_outage_line_2_inv
                )
            )
            phi_taylor_subset_list.append(phi_taylor_subset)
    phi_taylor = sum(phi_taylor_subset_list)
    return phi_taylor


def get_shapley_taylor_all_outage_lines(outage_line, B_d, M, I, L_inv=None):
    """
    Calculates shapley taylor values for all outage lines in M.
    Args:
        outage_line: Outage line for which the shapley taylor value with all other outage lines in M shall be calculated
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    """
    shapley_taylor_values = {}
    if L_inv == None:
        L_inv = get_L_inv(B_d, I)
    for outage_line_2 in M:
        shapley_taylor_value = get_shapley_taylor(
            outage_line_1=outage_line,
            outage_line_2=outage_line_2,
            B_d=B_d,
            M=M,
            I=I,
            L_inv=L_inv,
        )
        shapley_taylor_values[outage_line_2] = shapley_taylor_value
    return shapley_taylor_values


def check_shapley_taylor_sum_assumption(
    outage_line, B_d, M, I, L_inv=None, rtol=1e-3, atol=1e-5
):
    """
    Checks the identity that the shapley values are given by the sum of all possible shapley taylor indices divided by 2
    with the given outage line plus the direct effect of the line outage.

    Args:
        outage_line (int): The outage_line of the shapley value.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
        rtol (float): Relative tolerance for differences.
        atol (float): Absolute tolerance for differences.

    Returns:
        bool: True if Shapley values and Shapley taylor sum are close, else False.
    """
    shapley_value = get_shapley_value(
        outage_line=outage_line, B_d=B_d, M=M, I=I, L_inv=L_inv
    )
    shapley_taylor_indices = {}
    for second_outage_line in M:
        if second_outage_line is not outage_line:
            shapley_taylor_indices[second_outage_line] = get_shapley_taylor(
                outage_line_1=outage_line,
                outage_line_2=second_outage_line,
                B_d=B_d,
                M=M,
                I=I,
                L_inv=L_inv,
            )
    shapley_taylor_indices[outage_line] = get_direct_effect(
        outage_line=outage_line, B_d=B_d, I=I, L_inv=L_inv
    )
    shapley_taylor_sum = 0.5 * sum(shapley_taylor_indices.values())
    if np.allclose(shapley_value, shapley_taylor_sum, rtol=rtol, atol=atol):
        print(
            f"Shapley values and 0.5 Shapley taylor sum are close enough (rtol={rtol}, atol={atol})."
        )
        return True
    else:
        print(
            f"Shapley values and 0.5 Shapley taylor sum are not close (rtol={rtol}, atol={atol})."
        )
        print(shapley_taylor_sum)
        print(shapley_value)
        return False


def get_flow_attribution_shapley_taylor(
    affected_line, outage_line_1, outage_line_2, P, B_d, M, I, L_inv=None
):
    """
    Get the Shapley-taylor attribution for a specific flow on a line in a list of lines.

    Args:
        affected_line (int): The affected line number to get the attribution for.
        outage_line_1 (int): The first outage line number.
        outage_line_2 (int): The second outage line number.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        float: The Shapley-taylor attribution for the specified flow on the affected line.
    """
    phi_taylor = get_shapley_taylor(
        outage_line_1=outage_line_1,
        outage_line_2=outage_line_2,
        B_d=B_d,
        M=M,
        I=I,
        L_inv=L_inv,
    )
    phi_taylor_affected_line = flow_attribution(
        affected_line=affected_line, P=P, B_d=B_d, I=I, value=phi_taylor
    )
    return phi_taylor_affected_line


def get_flow_attribution_shapley_taylor_all_outage_lines(
    affected_line, P, B_d, M, I, L_inv=None
):
    """
    Get the flow attribution Shapley-taylor value for all lines in a list of failed lines for one affected line.

    Args:
        affected_line (int): The affected line number to get the attribution for.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        dict: A dictionary with line indices as keys and their Shapley-taylor values as values.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    shapley_taylor_values = {}
    outage_lines = list(M)
    for outage_line_1, outage_line_2 in tqdm(
        it.combinations(outage_lines, 2),
        total=len(list(it.combinations(outage_lines, 2))),
    ):
        shapley_taylor_values[(outage_line_1, outage_line_2)] = (
            get_flow_attribution_shapley_taylor(
                affected_line=affected_line,
                outage_line_1=outage_line_1,
                outage_line_2=outage_line_2,
                P=P,
                B_d=B_d,
                M=M,
                I=I,
                L_inv=L_inv,
            )
        )
    return shapley_taylor_values


def get_shapley_taylor_all_outage_lines(B_d, M, I, L_inv=None):
    """
    Get the Shapley-taylor value for all lines in a list of failed lines.

    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        dict: A dictionary with line indices as keys and their Shapley-taylor values as values.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    phi_taylor_values = {}
    outage_lines = list(M)
    partial_func = partial(get_shapley_taylor, B_d=B_d, M=M, I=I, L_inv=L_inv)
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(partial_func, outage_line_1, outage_line_2): (
                outage_line_1,
                outage_line_2,
            )
            for outage_line_1, outage_line_2 in it.combinations(outage_lines, 2)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            outage_line_tuple = futures[future]
            phi_taylor_values[outage_line_tuple] = future.result()
    return phi_taylor_values


def get_flow_attribution_shapley_taylor_all_outage_and_affected_lines(
    affected_lines, P, B_d, M, I, L_inv=None
):
    """
    Get the flow attribution Shapley-taylor value for all lines in a list of failed lines for all affected lines in a list.

    Args:
        affected_lines (int list): list of affected lines.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        dict: A dictionary with line indices as keys and their Shapley-taylor values as values.
    """
    phi_taylor_values = get_shapley_taylor_all_outage_lines(
        B_d=B_d, M=M, I=I, L_inv=L_inv
    )
    shap_taylor_values = flow_attribution_all_affected_lines_dict(
        affected_lines=affected_lines, P=P, B_d=B_d, I=I, value_dict=phi_taylor_values
    )
    return shap_taylor_values


def get_flow_attribution_shapley_taylor_all_affected_lines(
    affected_lines, outage_line_1, outage_line_2, P, B_d, M, I, L_inv=None
):
    """
    Get the Shapley-taylor attribution for all lines in a list of affected lines.

    Args:
        affected_lines (int list): list of affected lines.
        outage_line_1 (int): The first outage line number.
        outage_line_2 (int): The second outage line number.
        P (float): Power flow on the affected line.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        dict: A dictionary with line indices as keys and their Shapley-taylor attribution as values.
    """
    shapley_taylor_values = {}
    phi_taylor = get_shapley_taylor(
        outage_line_1=outage_line_1,
        outage_line_2=outage_line_2,
        B_d=B_d,
        M=M,
        I=I,
        L_inv=L_inv,
    )
    for affected_line in affected_lines:
        shapley_taylor_values[affected_line] = flow_attribution(
            affected_line=affected_line, P=P, B_d=B_d, I=I, value=phi_taylor
        )
    return shapley_taylor_values


# Shap value knn approx
def get_shapley_value_knn_approx(outage_line, B_d, M, I, knn_dict, L_inv=None):
    """
    Get the shapley value approximation for a specific line in a list of lines.

    Args:
        outage_line (int): The outage_line number to get the attribution for.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        cluster_dict (dict): dict with cluster index for each line
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        float: The shapley value approximation for the specified outage_line.
    """

    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    # Iterate over all subset lengths
    if outage_line in M:
        M_new = M - {outage_line}
    else:
        raise ValueError(f"Element '{outage_line}' not found in set M")
    phi_subset_list = []
    knn_lines = knn_dict[outage_line]
    # Lines in other cluster as outage line
    non_knn_lines = [line for line in M_new if line not in knn_lines]

    # Only iterate over
    for subset_length in range(len(knn_lines) + 1):
        for subset in it.combinations(knn_lines, subset_length):
            # Calculate the weighting factor
            W = 0
            for non_cluster_subset_length in range(len(non_knn_lines) + 1):
                comb = math.comb(len(non_knn_lines), non_cluster_subset_length)
                W += weight_factor(subset_length + non_cluster_subset_length, M) * comb

            if subset_length == 0:
                L_subset_inv = L_inv
            else:
                L_subset_inv = get_L_subset_inv(subset, L_inv, B_d, I)

            # Calculate for subset with outage_line removed
            L_subset_outage_line_inv = get_L_subset_inv(
                subset + (outage_line,), L_inv, B_d, I
            )
            # Calculate phi for this subset
            phi_subset = W * (L_subset_outage_line_inv - L_subset_inv)
            phi_subset_list.append(phi_subset)
    phi = sum(phi_subset_list)

    return phi


def get_shapley_value_knn_approx_all_outage_lines(B_d, M, I, knn_dict, L_inv=None):
    """
    Calculates approximate shapley value for all outage lines using multiprocessing.
    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        knn_dict (dict): dict with knn index for each line
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        dict: The shapley value for all outages with outage lines as keys
    """
    shapley_values = {}
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    # Partially apply arguments other than outage_line
    partial_func = partial(
        get_shapley_value_knn_approx,
        B_d=B_d,
        M=M,
        I=I,
        knn_dict=knn_dict,
        L_inv=L_inv,
    )

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(partial_func, outage_line): outage_line for outage_line in M
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            outage_line = futures[future]
            shapley_values[outage_line] = future.result()

    return shapley_values


# Shap Taylor knn approx
def get_shapley_taylor_knn_approx(
    outage_line_1, outage_line_2, B_d, M, I, knn_dict, L_inv=None
):
    """
    Get the Shapley-taylor attribution approximation for two outage lines.

    Args:
        outage_line_1 (int): The first outage line number.
        outage_line_2 (int): The second outage line number.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        knn_dict (dict): dict with knn index for each line
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        float: The Shapley-taylor attribution approximation for the two outage lines.
    """

    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    # Iterate over all subset lengths
    if outage_line_1 in M and outage_line_2 in M:
        M_new = M - {outage_line_1, outage_line_2}
    else:
        raise ValueError(f"Element '{outage_line_1, outage_line_2}' not found in set M")
    phi_taylor_subset_list = []
    if (outage_line_1, outage_line_2) in knn_dict:
        knn_lines = knn_dict[(outage_line_1, outage_line_2)]
    elif (outage_line_2, outage_line_1) in knn_dict:
        knn_lines = knn_dict[(outage_line_2, outage_line_1)]
    cluster_lines = [line for line in M_new if line in knn_lines]
    # Lines in other cluster as outage line
    non_cluster_lines = [line for line in M_new if line not in cluster_lines]
    # Only iterate over
    for subset_length in range(len(cluster_lines) + 1):
        for subset in it.combinations(cluster_lines, subset_length):
            # Calculate the weighting factor
            W = 0
            for non_cluster_subset_length in range(len(non_cluster_lines) + 1):
                comb = math.comb(len(non_cluster_lines), non_cluster_subset_length)
                W += weight_factor(subset_length + non_cluster_subset_length, M) * comb

            if subset_length == 0:
                L_subset_inv = L_inv
            else:
                L_subset_inv = get_L_subset_inv(subset, L_inv, B_d, I)
            # Calculate for subset with outage_line added
            L_subset_outage_line_1_inv = get_L_subset_inv(
                subset + (outage_line_1,), L_inv, B_d, I
            )
            L_subset_outage_line_2_inv = get_L_subset_inv(
                subset + (outage_line_2,), L_inv, B_d, I
            )
            L_subset_outage_lines_inv = get_L_subset_inv(
                subset + (outage_line_1, outage_line_2), L_inv, B_d, I
            )
            phi_taylor_subset = (
                2
                * W
                * (
                    L_subset_outage_lines_inv
                    + L_subset_inv
                    - L_subset_outage_line_1_inv
                    - L_subset_outage_line_2_inv
                )
            )
            phi_taylor_subset_list.append(phi_taylor_subset)
    phi_taylor = sum(phi_taylor_subset_list)

    return phi_taylor


def get_shapley_taylor_knn_approx_all_outage_lines(B_d, M, I, knn_dict, L_inv=None):
    """
    Get the Shapley-taylor approx value for all lines in a list of failed lines.

    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        knn_dict (dict): dict with knn index for each line
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.

    Returns:
        dict: A dictionary with line indices as keys and their Shapley-taylor values as values.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    phi_taylor_values = {}
    outage_lines = list(M)

    partial_func = partial(
        get_shapley_taylor_knn_approx,
        B_d=B_d,
        M=M,
        I=I,
        knn_dict=knn_dict,
        L_inv=L_inv,
    )
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(partial_func, outage_line_1, outage_line_2): (
                outage_line_1,
                outage_line_2,
            )
            for outage_line_1, outage_line_2 in it.combinations(outage_lines, 2)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            outage_line_tuple = futures[future]
            phi_taylor_values[outage_line_tuple] = future.result()

    return phi_taylor_values


# Higher order
def discrete_derivative(line_1, B_d, M, I, L_inv=None):
    """
    Calculates the discrete derivative.

    Args:
        line_1 (int): Line for which the derivative should be calculated.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    if not M:
        L_inv_M = L_inv
    else:
        L_inv_M = get_L_subset_inv(subset=M, L_inv=L_inv, B_d=B_d, I=I)
    L_inv_M_line = get_L_subset_inv(subset=M | {line_1}, L_inv=L_inv, B_d=B_d, I=I)
    delta = L_inv_M_line - L_inv_M

    return delta


def get_higher_order_derivative(lines, order, B_d, M, I, L_inv=None):
    """
    Calculates higher order discrete derivates recursevely.

    Args:
        lines (list): lines for that the higher order derivative should be calculated
        order (int): order of the derivative. Should be the order of lines.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.


    Returns:
        matrix: higher order derivative
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    M -= set(lines)
    if order != len(lines):
        print(f"Derivative order {order} does not match the number of lines {lines}.")
        return -1
    else:
        if order == 2:
            delta_M = discrete_derivative(
                line_1=lines[1], M=M, B_d=B_d, I=I, L_inv=L_inv
            )
            delta_M_line = discrete_derivative(
                line_1=lines[1], M=M | {lines[0]}, B_d=B_d, I=I, L_inv=L_inv
            )
            return delta_M_line - delta_M
        else:
            line = lines[0]
            lines_new = lines[1:]
            order_new = order - 1
            delta_higher_M = get_higher_order_derivative(
                lines=lines_new, order=order_new, B_d=B_d, M=M, I=I, L_inv=L_inv
            )
            delta_higher_M_line = get_higher_order_derivative(
                lines=lines_new,
                order=order_new,
                B_d=B_d,
                M=M | {line},
                I=I,
                L_inv=L_inv,
            )
            return delta_higher_M_line - delta_higher_M


def get_flow_attribution_higher_order_derivative(
    affected_line, P, lines, order, B_d, M, I, L_inv=None
):
    """
    Flow attribution for higher order derivative for a affected line.
    Args:
        affected_line (int): The affected line number to get the attribution for.
        P (float): Power flow on the affected line.
        lines (list): lines for that the higher order derivative should be calculated
        order (int): order of the derivative. Should be the order of lines.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
    Returns:
        float: flow attribution of higher order derivative for affected line
    """
    higher_order_derivative = get_higher_order_derivative(
        lines=lines, order=order, B_d=B_d, M=M, I=I, L_inv=L_inv
    )
    flow_attribution_higher_order_derivative = flow_attribution(
        affected_line=affected_line, P=P, B_d=B_d, I=I, value=higher_order_derivative
    )
    return flow_attribution_higher_order_derivative


def get_flow_attribution_higher_order_derivative_all_affected_lines(
    affected_lines, P, lines, order, B_d, M, I, L_inv=None
):
    """
    Flow attribution of higher order derivative for all affected lines in a list of affected lines.
    Args:
        affected_lines (int list): list of affected lines.
        P (float): Power flow on the affected line.
        lines (list): lines for that the higher order derivative should be calculated
        order (int): order of the derivative. Should be the order of lines.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
    Returns:
        dict: with affected lines as key and flow attributions of higher order derivatives as values.
    """
    if any(line in affected_lines for line in lines):
        affected_lines = list(set(affected_lines) - set(lines))
        print(
            "Removed some lines from affected lines. As the lines should not be part of outage lines."
        )
    higher_order_derivative = get_higher_order_derivative(
        lines=lines, order=order, B_d=B_d, M=M, I=I, L_inv=L_inv
    )
    flow_attribution_higher_order_derivative = flow_attribution_all_affected_lines(
        affected_lines=affected_lines, P=P, B_d=B_d, I=I, value=higher_order_derivative
    )
    return flow_attribution_higher_order_derivative


def higher_order_shap_taylor(outage_lines, B_d, M, I, L_inv=None):
    """
    Calculates higher order shap taylor index with |S|=k.

    Args:
        outage_lines (list int): list of outage lines, also determined order of shap taylor
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
    Return:
        phi_taylor (matrix): shap taylor matrix
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    # Iterate over all subset lengths
    M_new = M - set(outage_lines)
    if not set(outage_lines).issubset(M):
        raise ValueError(f"Elements '{outage_lines}' not found in set M")
    phi_taylor_subset_list = []
    for subset_length in range(len(M_new) + 1):
        for subset in it.combinations(M_new, subset_length):
            W = weight_factor(subset_length, M) * len(
                outage_lines
            )  # Should W be multiplied with factor corresponding to len outage_lines?
            derivative = get_higher_order_derivative(
                lines=outage_lines,
                order=len(outage_lines),
                B_d=B_d,
                M=set(subset),
                I=I,
                L_inv=L_inv,
            )
            phi_taylor_subset = W * derivative
            phi_taylor_subset_list.append(phi_taylor_subset)
    phi_taylor = sum(phi_taylor_subset_list)
    return phi_taylor


def get_all_direct_effects(B_d, M, I, L_inv=None):
    """
    Calculates direct effects for all possible outage line in M and saves this in a dict.
    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
    Returns:
        direct_effects (dict): Direct effects of all outage lines, with outage line as key.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    direct_effects = {}
    for outage_line in M:
        direct_effects[outage_line] = get_direct_effect(
            outage_line=outage_line, B_d=B_d, I=I, L_inv=L_inv
        )
    return direct_effects


def get_all_second_order_effects(B_d, M, I, L_inv=None):
    """
    Calculates second order effects for all possible outage line combinations in M and saves this in a dict.
    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
    Returns:
        second_order_effects (dict): Dict of second order effects, with keys being tuples of corresponding outage lines.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    second_order_effects = {}
    for outage_line_1, outage_line_2 in it.combinations(M, 2):
        second_order_effects[(outage_line_1, outage_line_2)] = (
            get_higher_order_derivative(
                lines=[outage_line_1, outage_line_2],
                order=2,
                B_d=B_d,
                M=set([outage_line_1, outage_line_2]),
                I=I,
                L_inv=L_inv,
            )
        )
    return second_order_effects


def compute_combo(combo, B_d, M, I, L_inv):
    combo_list = tuple(combo)
    value = higher_order_shap_taylor(
        outage_lines=list(combo), B_d=B_d, M=M, I=I, L_inv=L_inv
    )
    return value


def get_all_third_order_shap_taylor(B_d, M, I, L_inv=None):
    """
    Calculates all third order shap taylor indices from outage line combinations and saves them in a dict.
    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
    Returns:
        third_order_shap (dict): Dict of third order shap taylor indices. The keys are a touple of all corresponding outage lines.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    combos = it.combinations(M, 3)
    worker = partial(compute_combo, B_d=B_d, M=M, I=I, L_inv=L_inv)
    third_order_shap = {}
    with ProcessPoolExecutor(max_workers=1) as executor:
        # submit all jobs
        futures = {executor.submit(worker, combo): combo for combo in combos}

        # iterate as they complete with progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            combo_list = futures[future]
            value = future.result()
            third_order_shap[tuple(combo_list)] = value
    return third_order_shap


def third_order_shap_terms(B_d, M, I, L_inv=None):
    """
    Calculates all terms of a third order shapley taylor expansion.
    Args:
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
    Returns:
        direct_effects (dict): Dict of all direct effects. The keys are the outage lines.
        second_order_effects): Dict of all second order effect. The keys are the outage lines.
        third_order_shap (dict): Dict of all third order shap taylor indices including correction terms. The keys are the outage lines.
    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    direct_effects = get_all_direct_effects(B_d=B_d, M=M, I=I, L_inv=L_inv)
    second_order_effects = get_all_second_order_effects(B_d=B_d, M=M, I=I, L_inv=L_inv)
    third_order_shap = get_all_third_order_shap_taylor(B_d=B_d, M=M, I=I, L_inv=L_inv)
    return direct_effects, second_order_effects, third_order_shap


def get_flow_attribution_third_order_shap_terms_all_affected_line(
    affected_lines, B_d, M, P, I, L_inv=None, file_path=None
):
    """
    Calculates flow attribution of all third order shap taylor indices from outage line combinations and saves them in a dict.
    Args:
        affected_lines (int list): list of affected lines.
        B_d (sparse matrix): Diagonal matrix with susceptance on diagonal.
        M (int set): Set of failed lines.
        P (float): Power flow on the affected line.
        I (sparse matrix): Incidence matrix
        L_inv (matrix): Inverse Laplacian matrix. If None, it will be computed from B_d.
        file_path (string): file path where to save the flow attribution. Default is None, which results in no saving.
    Returns:
        values (dict): Flow attribution dict.

    """
    if L_inv is None:
        L_inv = get_L_inv(B_d, I)
    values = []
    direct_effects, second_order_effects, third_order_shap = third_order_shap_terms(
        B_d=B_d, M=M, I=I, L_inv=L_inv
    )
    for affected_line in affected_lines:
        direct_effect_attribution = {}
        second_order_effects_attribution = {}
        third_order_shap_attribution = {}
        for key in direct_effects:
            direct_effect_attribution[key] = flow_attribution(
                affected_line=affected_line,
                P=P,
                B_d=B_d,
                I=I,
                value=direct_effects[key],
            )
        for key in second_order_effects:
            second_order_effects_attribution[key] = flow_attribution(
                affected_line=affected_line,
                P=P,
                B_d=B_d,
                I=I,
                value=second_order_effects[key],
            )
        for key in third_order_shap:
            third_order_shap_attribution[key] = flow_attribution(
                affected_line=affected_line,
                P=P,
                B_d=B_d,
                I=I,
                value=third_order_shap[key],
            )
        values.append(
            {
                "affected_line": affected_line,
                "direct_effects": direct_effect_attribution,
                "second_order_effects": second_order_effects_attribution,
                "third_order_shap": third_order_shap_attribution,
            }
        )
        total_sum = (
            sum(direct_effect_attribution.values())
            + sum(second_order_effects_attribution.values())
            + sum(third_order_shap_attribution.values())
        )
        print("Total_sum: ", total_sum)
        print(
            "Power flow: ",
            get_real_power_flow_change(
                affected_line=affected_line, P=P, B_d=B_d, M=M, I=I
            ),
        )
    if file_path is not None:
        with open(file_path, "wb") as f:
            pkl.dump(values, f)
    return values
