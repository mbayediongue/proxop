"""
Version : 1.0 ( 06-13-2022).

DEPENDENCIES:
     - 'prox_svd.py' located in the folder 'utils'
     - 'fun_svd.py' located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.prox_svd import prox_svd
from proxop.utils.fun_svd import fun_svd


class NuclearNormRidge:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:


            f(x) = (1/2)*||X||_F^2 + mu*R1(C) =(1/2)||S||_2^2 + mu*||s||_1

    where X = U*diag(s)*V.T \in R^{M*N}  (Singular Value decomposition)

    'gamma' is the scale factor

     INPUTS
    ========
     x         -  (M,N) -array_like ( representing an M*N matrix )
     mu        - positive scalar [default: mu=1]
     gamma     - positive scalar  [default: gamma=1]
    """

    def __init__(
            self,
            mu: float = 1,
            gamma: float = 1
    ):
        if np.any(gamma <= 0) or np.size(gamma) > 1:
            raise Exception("'gamma'  must be a strictly positive scalar")
        if np.any(mu <= 0) or np.size(mu) > 1:
            raise Exception("'mu'  must be a strictly positive scalar")
        self.gamma = gamma
        self.mu = mu

    def prox(self, x: np.ndarray) -> np.ndarray:
        mu = self.mu

        def prox_phi(s, gam_):
            a = -mu / (1 + gam_)
            b = mu / (1 + gam_)
            ss = s / (1 + gam_)
            phi_s = np.minimum(0, ss - a * gam_) + np.maximum(0, ss - b * gam_)
            return phi_s

        return prox_svd(x, self.gamma, prox_phi)

    def __call__(self, x: np.ndarray) -> float:
        def fun_phi(s):
            return 0.5 * np.sum(s ** 2) + self.mu * np.sum(np.abs(s))

        return self.gamma * fun_svd(x, 1, fun_phi)

    def _check(self, x):
        if len(np.shape(x)) != 2:
            raise ValueError(
                "'x' must be an (M,N) -array_like ( representing a M*N matrix )"
            )
