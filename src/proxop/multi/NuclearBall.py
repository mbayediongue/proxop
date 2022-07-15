"""
Version : 1.0 ( 06-21-2022).

DEPENDENCIES:
     - 'AbsValue.py' located in the folder 'scalar'
     - 'prox_svd.py' located in the folder 'utils'
     - 'NuclearNorm.py' located in the folder 'multi'
     - 'Linf.py' located in the folder 'multi'
     - 'Max.py' in the folder 'multi'
     - 'L1Ball.py' in the folder 'indicator'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.indicator.L1Ball import L1Ball
from proxop.utils.prox_svd import prox_svd
from proxop.multi.NuclearNorm import NuclearNorm


class NuclearBall:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the indicator of the constraint set:

                         ||X||_N =||s||_1 <= eta   (nuclear norm ball constraint)

            where X = U*diag(s)*V.T \in R^{M*N}

     INPUTS
    ========
     x         -  (M,N) -array_like ( representing an M*N matrix )
     gamma     - positive, scalar or ND array compatible with the size of 'x'
                 [default: gamma=1]
     eta       - positive, scalar or ND array compatible with the size of 'x'
    """

    def __init__(
            self,
            eta: float or np.ndarray,
            gamma: float or np.ndarray = 1
    ):
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it is an array)" +
                " must be strictly positive"
            )
        self.gamma = gamma
        self.eta = eta

    def prox(self, x: np.ndarray) -> np.ndarray:
        return prox_svd(x, 1, L1Ball(eta=self.gamma).prox)

    def __call__(self, x: np.ndarray) -> float:
        nuclear_norm = NuclearNorm(gamma=self.gamma)(x)
        if nuclear_norm <= self.eta:
            return 0
        return np.inf
