"""
Version : 1.0 ( 06-20-2022).

DEPENDENCIES:
    - 'Max.py' located in the folder 'multi'
    - 'Linf.py' located in the folder 'multi'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.multi.Linf import Linf


class Linf1:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is a sum of infinitive norms:

                         f(x) = \sum_{i=1}^N \sup_{1\le j \le M} |X_{i,j}|

            where X = U*diag(s)*V.T \in R^{M*N}  (Singular Value decomposition)

    'gamma' is the scale factor

     INPUTS
    ========
     x         -  (M,N) -array_like ( representing an M*N matrix )
     gamma     - positive, scalar or ND array compatible with the size of 'x'
                 [default: gamma=1]
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it is an array)" +
                " must be strictly positive"
            )
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        return Linf(gamma=self.gamma, axis=1).prox(x)

    def __call__(self, x: np.ndarray) -> float:
        return Linf(gamma=self.gamma, axis=1)(x)
