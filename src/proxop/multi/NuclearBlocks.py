"""
Version : 1.0 ( 06-20-2022).

DEPENDENCIES:
     - 'NuclearNorm.py' located in the folder 'multi'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""


import numpy as np
from proxop.multi.NuclearNorm import NuclearNorm


class NuclearBlocks:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:


                f(x) = gamma * \sum_{i= 1}^N \sum_{j=1}^M w_{i,j} \|X_{i,j}\|_N

            where X = U*diag(s)*V.T \in R^{M*N}

     INPUTS
    ========
     x        -  (M,N) -array_like ( representing an M*N matrix )

     ind_r    - Vector partitioning the rows of X in groups EXAMPLE: ind_r=[1 2 2 3 3 1]
     means there are max(ind_r)=3 blocks:  the first block contains the first and last
     rows of x, the 2nd block contains the 2nd and the third row, the third block
     contains the 4th and the 5th row.

     ind_c    - Vector partitioning the columns of X in groups
     W        - positive ND array of size (max(ind_r) , max(ind_c) )
     gamma    - positive, scalar or ND array compatible with the size of 'x'
                 [default: gamma=1]
    """

    def __init__(
            self,
            w: float or np.ndarray,
            ind_r: np.ndarray,
            ind_c: np.ndarray,
            gamma: float or np.ndarray = 1
            ):
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it is an array)" +
                " must be strictly positive"
            )
        if np.any(w <= 0) or np.any((np.max(ind_r), np.max(ind_c)) != np.shape(w)):
            raise Exception(
                "'W' must be positive and of shape '( max(ind_r), max(ind_c))'"
            )
        self.gamma = gamma
        self.w = w
        self.ind_r = np.array(ind_r)
        self.ind_c = np.array(ind_c)

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        scale = self.gamma
        ind_r = self.ind_r
        ind_c = self.ind_c
        w = self.w

        m, n = np.shape(x)
        if np.size(ind_r) != m or np.size(ind_c) != n:
            raise ValueError("'ind_r' or 'ind_c' has not the right length: " +
                             "It is expected: (len(ind_r), len(ind_c)) = np.shape(x)")
        prox_x = np.zeros((m, n))
        for i in range(max(ind_r)):
            for j in range(max(ind_c)):
                ind_i = ind_r == i
                ind_j = ind_c == j
                mask = np.dot(ind_i.T, ind_j)
                # The elements that don't belong to the block are replaced by zeros
                x_mask = x * mask

                # Update of p
                prox_x = prox_x + NuclearNorm(gamma=w[i, j] * scale).prox(x_mask)

        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        ind_r = self.ind_r
        ind_c = self.ind_c
        w = self.w
        fun_x = 0
        for i in range(max(ind_r)):
            for j in range(max(ind_c)):
                ind_i = ind_r == i
                ind_j = ind_c == j
                mask = np.dot(ind_i.T, ind_j)
                # The elements that don't belong to the block are replaced by zeros
                x_mask = x * mask
                # Update of fun_x
                fun_x = fun_x + NuclearNorm(gamma=w[i, j] * self.gamma)(x_mask)
        return fun_x

    def _check(self, x):
        if len(np.shape(x)) != 2:
            raise ValueError(
                "'x' must be an (M,N) -array_like ( representing a M*N matrix )"
            )
