#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
Purely matrix-based simulation of cascading failures in power grids
"""

import itertools

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# First column: num_parallel before failure
# Second column: num_parallel after failure
## Not in original data and exists due to reduction from num_parallel > 3
## 0.59210526, 0.88157895, 1.18421053, 1.47368421, 1.59210526,
#  1.77631579, 2.06578947, 2.15789474, 2.18421053, 2.31578947, 2.36842105,
#  2.44736842, 2.57894737, 2.59210526, 2.73684211, 2.77631579
# 0.59210526 provides path to zero
LOOKUP_TABLE_NP = np.array(
    [
        [0.28947368, 0.0],
        [0.57894737, 0.28947368],
        [0.59210526, 0.0],
        [0.86842105, 0.57894737],
        [0.88157895, 0.28947368],
        [1.0, 0.0],
        [1.15789474, 0.86842105],
        [1.18421053, 0.59210526],
        [1.28947368, 0.28947368],
        [1.44736842, 1.15789474],
        [1.47368421, 0.88157895],
        [1.57894737, 0.57894737],
        [1.59210526, 0.59210526],
        [1.73684211, 1.44736842],
        [1.77631579, 1.18421053],
        [1.86842105, 0.86842105],
        [2.0, 1.0],
        [2.02631579, 1.73684211],
        [2.06578947, 1.47368421],
        [2.15789474, 1.15789474],
        [2.18421053, 1.18421053],
        [2.28947368, 1.28947368],
        [2.31578947, 2.02631579],
        [2.36842105, 1.77631579],
        [2.44736842, 1.44736842],
        [2.57894737, 1.57894737],
        [2.59210526, 1.59210526],
        [2.65789474, 2.06578947],
        [2.73684211, 1.73684211],
        [2.77631579, 1.77631579],
        [2.86842105, 1.86842105],
    ]
)

# ! Remove after testing
# for line extension extend lookup table:
LOOKUP_TABLE_NP = np.concatenate((LOOKUP_TABLE_NP, LOOKUP_TABLE_NP + 1), axis=0)

# The smallest cable seems to be 0.3351.. here instead of 0.2894.. as for sclopf
## Decision to reduce 2.34 by 1 and not be 0.3351...
LOOKUP_TABLE_NP_non_sclopf = np.array(
    [
        [0.33518006, 0.0],
        [0.67036011, 0.33518006],
        [0.67590027, 0.0],
        [1.0, 0.0],
        [1.00554017, 0.67036011],
        [1.01108033, 0.67590027],
        [1.33518006, 1.0],
        [1.34072022, 1.00554017],
        [1.34626039, 1.01108033],
        [1.67036011, 1.33518006],
        [1.67590028, 1.34072022],
        [1.68144040, 1.34626039],
        [2.0, 1.0],
        [2.00554017, 1.00554017],
        [2.01108033, 1.67590028],
        [2.01662050, 1.68144040],
        [2.33518006, 1.33518006],
        [2.34072022, 1.34072022],
        [2.34626039, 2.01108033],
        [2.67036011, 1.67036011],
        [2.67590028, 1.67590028],
        [2.68144044, 2.34626039],
    ]
)

# have to be added due to >3 being reduced by one. This are only the >2 ones
# add = [2.00554017, 2.01108033, 2.0166205, 2.33518006, 2.34072022, 2.34626039,
#       2.67036011, 2.67590028, 2.68144044] and 1.68144040, 0.67590027 is new smallest line
# not appearing in PyPSA network


def calc_num_parallel_after_failure(num_parallel: float, use_sclopf: bool = True):
    """Calculate the new effective number of circuits on a line
    after removing one circuit. The new value depends on the line type
    and is indicated in a lookup table.

    Args:
        num_parallel (float): Old value of effective number of circuits
        use_sclopf (bool): Decides which look up table to use, since sclopf and lopf PyPSA
        networks have different num_parallel value giving an effective line value.

    Returns:
        float: new value
    """

    if use_sclopf:
        look_up_table = LOOKUP_TABLE_NP
    else:
        look_up_table = LOOKUP_TABLE_NP#LOOKUP_TABLE_NP_non_sclopf

    assert num_parallel >= 1e-8, (
        "Line removal for num_parallel=0 not correct."
        + " Line was either already removed a wrong num_parallel"
        + " was assigned."
    )

    if 0 < num_parallel < 3:
        try:
      

            
            num_parallel_case = np.argwhere(np.isclose(look_up_table[:, 0], num_parallel))[
                0, 0
            ]
            
            
            num_parallel_new = look_up_table[num_parallel_case, 1]
            
        except IndexError:
           raise LookupError(f"Num_parallel before failure '{num_parallel}' is not in Lookup table.")

    elif num_parallel >= 3:
        num_parallel_new = num_parallel - 1

    else:
        raise ValueError("num_parallel does not have a valid value!")

    return num_parallel_new


def remove_line_from_Bd(
    B_d_in,
    num_parallel_ls,
    line_limits_ls,
    del_idx,
    remove_all_circuits=False,
    use_sclopf: bool = True,
):
    """Remove a line by changing the susceptances, the number of parallel lines and
    the line limits.

    Args:
        B_d_in (sparse diagonal matrix): Diagonal matrix with susceptances
        num_parallel_ls (list): number quantifying the effective number of parrallel circuits on a line
        line_limit_ls (list): List of line limits that will be modified.
        del_idx (idx of ): idx of edge in graph that will be modified due to overloaded power line
        remove_all_circuits (bool): If False, remove only one circuit from the line.
        If True, remove the whole line with all circuits.
        use_sclopf (bool): If 'True' use num_parallel lookup table for non sclopf PyPSA network.
        
    Returns:
        tuple: Updated susceptance matrix, number of parallel lines and line limits
        
    """

    num_parallel_ls_after = num_parallel_ls.copy()
    line_limits_ls_after = line_limits_ls.copy()
    B_d_in_after = B_d_in.copy()
    
    
    # Select num_parallel of removed link
    num_parallel = num_parallel_ls[del_idx]
    assert num_parallel >= 1e-8, (
        "Line removal for num_parallel=0 not correct."
        + " Line was either already removed a wrong num_parallel was assigned. "
    )

    # Calculate new num_parallel after removal
    if remove_all_circuits:
        num_parallel_new = 0
    else:
        num_parallel_new = calc_num_parallel_after_failure(
            num_parallel, use_sclopf=use_sclopf
        )

    # Adapt network parameters accordingly
    num_par_factor = num_parallel_new / num_parallel
    num_parallel_ls_after[del_idx] = num_parallel_new
    B_d_in_after[del_idx, del_idx] *= num_par_factor
    line_limits_ls_after[del_idx] *= num_par_factor

    return (
        B_d_in_after,
        num_parallel_ls_after,
        line_limits_ls_after,
    )


def solve_lpf(P, B_d, I, L=None):
    """Solve linear power flow.

    Args:
        P (1d numpy array): Power injections
        B_d (sparse matrix): Susceptance matrix
        I (sparse matrix): Incidence matrix
        L (sparse matrix): Laplacian matrix

    Returns:
        flows: Power flows
    """

    if L == None:
        L = I.dot(B_d).dot(I.T)

    theta = np.zeros(L.shape[0])
    theta[1:] = sparse.linalg.spsolve(L[1:, 1:], P[1:])

    flows = B_d.dot((I.T).dot(theta))

    return flows

