# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 2022

@author: mbaye diongue
"""
import numpy as np


def pava_python(x):
    """
    This function computes the sum:

       S(j,k,:) = sum( x(j:k,:) ) / (k-j+1)

    for every 'j' and 'k' in {1,...,N}, where N = size(x,1).
    """
    N = np.size(x, 0)
    res = np.zeros(np.size(x))
    for n in range(N):
        max_sum = -np.inf
        for j in range(n + 1):
            min_sum = np.inf
            for k in range(n, N):
                sum_xl = 1 / (k - j + 1) * np.sum(x[j : (k + 1)])
                if sum_xl < min_sum:
                    min_sum = sum_xl
            if min_sum > max_sum:
                max_sum = min_sum
        res[n] = max_sum

    return res
