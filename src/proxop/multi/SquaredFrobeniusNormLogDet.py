"""
Version : 1.0 ( 06-21-2022).

DEPENDENCIES:
     -'prox_svd.py' - located in the folder 'utils'
     -'fun_svd.py'  - located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.prox_svd import prox_svd
from proxop.utils.fun_svd import fun_svd


class SquaredFrobeniusNormLogDet:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the funtion definded as:

               / -log( det(X) ) + mu*||X||_F^2    if X is a symmetric positive
         f(x)=|                                   definite matrix
              \  + inf                            otherwise

    where

           * X = U*diag(s)*V.T \in R^{M*N}  the Singular Value decomposition of X

           * det(X)= prod_i(s_i)   is the determinant of the matrix X

           * ||X||_F^2 = ||s||_2^2  the squared of the Frobenius norm of X

          * 'gamma*tau' is the scale factor

     Note:
        No checking is performed to verify whether X is symmetric or not when computing
        the proximity operator with the method 'prox'. X is assumed to
        be symmetric.

     INPUTS
    ========
     x               -  (M,N) -array_like ( representing an M*N matrix)
     mu              - positive scalar  [default: mu=1]
     gamma           - positive scalar  [default: gamma=1]
    """

    def __init__(
            self,
            mu: float = 1,
            gamma: float = 1):

        if np.any(gamma <= 0) or np.size(gamma) > 1:
            raise Exception("'gamma'  must be a strictly positive scalar")
        if np.any(mu < 0) or np.size(mu) > 1:
            raise Exception("'mu'  must be a positive scalar")
        self.gamma = gamma
        self.mu = mu
        self.a = 1

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        mu = self.mu

        def prox_phi(s, gam):
            return (s + np.sqrt(s**2 + 4 * gam * (2 * mu * gam + 1))) / (
                2 * (2 * gam * mu + 1)
            )

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
            return -np.log(np.prod(s)) + self.mu * np.sum(s**2)

        return self.gamma * fun_svd(x, 1, fun_phi)

    def _check(self, x):
        if len(np.shape(x)) != 2:
            raise ValueError(
                "'x' must be an (M,N) -array_like ( representing an M*N matrix )"
            )
