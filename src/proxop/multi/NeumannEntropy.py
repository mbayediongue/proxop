"""
Version : 1.0 ( 06-21-2022).

DEPENDENCIES:
     - 'prox_svd.py' located in the folder 'utils'
     - 'prox_svd.py' located in the folder 'utils'
     - 'lambert_W.py' located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.prox_svd import prox_svd
from proxop.utils.fun_svd import fun_svd
from proxop.utils.lambert_W import lambert_W


class NeumannEntropy:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the Neumann's entropy function defined as:

               /  trace( Xlog(X))    if X is a symmetric positive matrix
         f(x)=|
              \  + inf                          otherwise

    where

           * X = U*diag(s)*V.T \in R^{M*N}  the Singular Value decomposition of X

          * tr(X)= \sum_{i=0}^{N-1}(s_i)   is the trace of the matrix X

          * 'gamma' is the scale factor

     Note:
        No checking is performed to verify whether X is symmetric or not when computing
        the proximity operator with the method 'prox'. X is assumed to be symmetric.

     INPUTS
    ========
     x               -  (M,N) -array_like ( representing an M*N matrix)
     gamma           - positive scalar  [default: gamma=1]
    """

    def __init__(self, gamma: float = 1):
        if np.any(gamma <= 0) or np.size(gamma) > 1:
            raise Exception("'gamma'  must be a strictly positive scalar")
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)

        def prox_phi(s, gam):
            phi_s = np.zeros_like(s)
            # we use a taylor approximation of the LambertW function to avoid
            # divergence when the input is "too big"
            u = np.log(gam) + s / gam - 1
            mask = u > 100

            phi_s[mask] = u[mask] - u[mask] / (1 + u[mask]) * np.log(u[mask])

            # we use the Lambert_W function otherwise
            mask = np.logical_not(mask)
            if np.size(gam) > 1:
                gam = gam[mask]
            phi_s[mask] = gam * lambert_W(gam * np.exp(s[mask] / gam - 1))

            return phi_s

        return prox_svd(x, self.gamma, prox_phi, hermitian=True)

    def __call__(self, x: np.ndarray) -> float:
        self._check(x)
        TOL = 1e-12
        # Check if the matrix is symmetric
        if not np.allclose(x, np.transpose(x)):
            return np.inf

        def fun_phi(s):
            if np.any(s <= TOL):
                return np.inf
            return np.sum(s * np.log(s))

        return self.gamma * fun_svd(x, 1, fun_phi)

    def _check(self, x):
        if len(np.shape(x)) != 2:
            raise ValueError(
                "'x' must be an (M,N) -array_like ( representing a M*N matrix )"
            )
