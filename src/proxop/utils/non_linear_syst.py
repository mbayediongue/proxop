# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 2022

@author: mbaye diongue
"""

import numpy as np


def non_linear_sys(x, gamma, axis=None):
    """
    Find the scalar s, solution of the system:

        sum_{n=1}^{n=N}( max(0, x))= lam

    Parameters
    ----------
    x : ND array

    gamma : scalar

    axis: direction of block-wise processing
    Returns
    -------
    gamma : scalar
    """

    # 0. Linearize
    sz = np.size(x)
    if axis is None:
        x = x[:]

    # 1. Order the column elements (sort in decreasing order)
    s = -np.sort(-x, axis=axis)

    # 2. Compute the partial sums: c(j) = ( sum(s(1:j)) - gamma ) / j

    c = (np.cumsum(s, axis=axis) - gamma) / np.cumsum(
        np.ones(np.size(s)), axis=axis
    )

    # 3. Find the index: n = max{ j \in {1,...,B} : s(j) > c(j) }

    mask = np.where(s > c)[0]
    n = np.max(mask, axis)

    # 4. Compute the prox
    p = x - c[n]

    # 5. Output is zero if 's(j) > c(j)' for all j
    p = p * (1 - np.all(mask, axis))

    # 6. revert back
    p = np.reshape(p, sz)

    return p
