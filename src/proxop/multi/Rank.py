"""
Version : 1.0 ( 06-20-2022).

DEPENDENCIES:
     - 'prox_svd.py' located in the folder 'utils'
     - 'prox_svd.py' located in the folder 'utils'
     - 'L0Norm.py' located in the folder 'nonconvex'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.nonconvex.L0Norm import L0Norm
from proxop.utils.prox_svd import prox_svd
from proxop.utils.fun_svd import fun_svd


class Rank:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:

                         f(x)= Rank(X) = ||s||_0

    where X = U*diag(s)*V.T \in R^{M*N} (the Singular Value decomposition of X)

     INPUTS
    ========
     x         -  (M,N) -array_like ( representing an M*N matrix )
     gamma     - positive, scalar or ND array compatible with the size of 'x'
                 [default: gamma=1]
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array)" +
                " must be strictly positive"
            )
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        return prox_svd(x, 1, L0Norm(gamma=self.gamma).prox)

    def __call__(self, x: np.ndarray) -> float:
        return fun_svd(x, 1, L0Norm(gamma=self.gamma))
