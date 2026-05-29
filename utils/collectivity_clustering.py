#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as _np
import networkx as _nx
import itertools as it

from utils import distribution_factors


def get_knn_dict_new(M, G, k, tol=1e-20):
    """Creates a dict with k outage lines with highest mutual LODF.
    But outages with a mutual LODF below a threshold (default 1e-20) are always ignored.
    """
    LODFs = distribution_factors.calculate_LODF_matrix(
        G, is_multigraph=G.is_multigraph()
    )
    mutual_LODFs = _np.multiply(LODFs, LODFs.T)
    knn_dict = {}
    if k >= len(M):
        raise RuntimeError("There need to be at least as many outages as k.")
    for i in M:
        poss_neighbours = {}
        for j in M:
            if i == j:
                continue
            weight = mutual_LODFs[i, j]
            if abs(weight) < tol:
                weight = 0
                continue
            if weight < 0:
                raise ZeroDivisionError(
                    f"Product of LODFs is negative and amounts to {weight}"
                )
            poss_neighbours[j] = _np.sqrt(abs(weight))
        sorted_neighbours = sorted(
            poss_neighbours.items(), key=lambda x: x[1], reverse=True
        )
        knn_dict[i] = [nbr[0] for nbr in sorted_neighbours[:k]]
    return knn_dict
