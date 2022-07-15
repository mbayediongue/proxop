"""
Version : 1.0 ( 06-21-2022).

DEPENDENCIES:
     -'prox_svd.py' - located in the folder 'utils'
     -'fun_svd.py'  - located in the folder 'utils'
     -'newton.py'  in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.prox_svd import prox_svd
from proxop.utils.fun_svd import fun_svd
from proxop.utils.newton import newton_


class SchattenPenaltyLogDet:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:

               / -log( det(X) ) +  R_p(X)^p   if X is a symmetric positive
         f(x)=|                               definite matrix
              \  + inf                        otherwise

    where

           * X = U*diag(s)*V.T \in R^{M*N}  the Singular Value decomposition of X

           * R_p(X)= ||s||_p^p  is the p-Schatten norm for matrices

           * det(X)= prod_i(s_i)    is the determinant of the matrix X

           * 'gamma*tau' is the scale factor

    Note: No checking is performed to verify whether X is symmetric or not when
    computing the proximity operator with the method 'prox'. X is assumed to be
    symmetric.

     INPUTS
    ========
     x               -  (M,N) -array_like ( representing an M*N matrix
     p               -  scalar in ]0, 1]
     mu              - positive scalar [default: mu=1]
     gamma           - positive scalar  [default: gamma=1]
    """

    def __init__(
            self,
            p: float,
            mu: float = 1,
            gamma: float = 1):

        if np.any(gamma <= 0) or np.size(gamma) > 1:
            raise Exception("'gamma'  must be a strictly positive scalar")
        if np.any(mu < 0) or np.size(mu) > 1:
            raise Exception("'mu'  must be a positive scalar")
        if np.any(p < 0) or np.any(p > 1) or np.size(p) > 1:
            raise Exception("'p'  must belong to ]0, 1] ")
        self.gamma = gamma
        self.mu = mu
        self.p = p

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        mu = self.mu
        p = self.p

        def prox_phi(s, gam):
            # Note: same computation as in 'EpiPower.py' for q=2
            def polynom_phi(t):
                return gam * mu * p * t**p + t**2 - s * t - gam

            def derive_polynom_phi(t):
                return p * gam * mu * p * t ** (p - 1) + 2 * t - s

            # starting point
            low = 1e-10
            root_init = np.abs(s)
            # Finding the root of the polynom with the Newton method
            d = newton_(
                polynom_phi, fp=derive_polynom_phi, x0=root_init, low=low, high=np.inf
            )
            return d

        return prox_svd(x, self.gamma, prox_phi, hermitian=True)

    def __call__(self, x: np.ndarray) -> float:
        self._check(x)
        tol = 1e-12
        p = self.p
        # Check if the matrix is symmetric
        if not np.allclose(x, np.transpose(x)):
            return np.inf

        def fun_phi(s):
            if np.any(s <= tol):
                return np.inf
            return -np.log(np.prod(s)) + self.mu * np.sum((np.abs(s)) ** p)

        return self.gamma * fun_svd(x, 1, fun_phi)

    def _check(self, x):
        if len(np.shape(x)) != 2:
            raise ValueError(
                "'x' must be an (M,N) -array_like ( representinf a M*N matrix )"
            )