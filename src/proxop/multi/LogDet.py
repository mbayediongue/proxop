"""
Version : 1.0 ( 06-21-2022).

DEPENDENCIES:
     - 'prox_svd.py' located in the folder 'utils'
     - 'prox_svd.py' located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

from proxop.utils.prox_svd import prox_svd
from proxop.utils.fun_svd import fun_svd
import numpy as np


class LogDet:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the logarithm of determinant function:

               /-log( det(X) ) = -log( prod(s))  if X is a symmetric positive
         f(x)=|                                  definite matrix
              \ + inf                            otherwise

    where
           * det(X) is the determinant of the matrix X

           * X = U*diag(s)*V.T \in R^{M*N}  the Singular Value decomposition of X

          * 'gamma*tau' is the scale factor

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
        is_hermitian = False
        if np.allclose(x, np.transpose(x)):
            is_hermitian = True

        def prox_phi(s, gam):
            return 0.5 * (s + np.sqrt(s**2 + 4 * gam))

        return prox_svd(x, self.gamma, prox_phi, hermitian=is_hermitian)

    def __call__(self, x: np.ndarray) -> float:
        self._check(x)
        TOL = 1e-20
        # Check if the matrix is symmetric
        if not np.allclose(x, np.transpose(x)):
            return np.inf

        def fun_phi(s):
            if np.any(s <= TOL):
                return np.inf
            return -np.log(np.prod(s))

        return self.gamma * fun_svd(x, 1, fun_phi)

    def _check(self, x):
        if len(np.shape(x)) != 2:
            raise ValueError(
                "'x' must be an (M,N) -array_like ( representing a M*N matrix )"
            )
