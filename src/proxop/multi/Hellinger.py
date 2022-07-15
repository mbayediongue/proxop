"""
Version : 1.0 ( 06-22-2022).

DEPENDENCIES:
    -'Ialpha.py'  - located in the folder 'multi'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.multi.Ialpha import Ialpha


class Hellinger:
    r"""Compute the proximity operator and the evaluation of gamma*D.

    Where D is the  Hellinger function defined as:


                  /  ( sqrt(x) - sqrt(y) )^2   if x >= 0 and y >= 0
        D(x,y) = |
                 \  + inf                    otherwise

    'gamma' is the scale factor

     ========
     INPUTS
    ========
    x        - scalar or ND array
    y        - scalar if 'x' is a scalar , ND array with the same size as 'x' otherwise
    gamma    - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.any(gamma <= 0):
            raise Exception("'gamma'  must be strictly positive")
        self.gamma = gamma

    def prox(self, x: np.ndarray, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(x)
        scale = self.gamma
        return Ialpha(alpha=2, gamma=2 * scale).prox(x, y)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if np.any(x < 0) or np.any(y < 0):
            return np.inf
        return np.sum(self.gamma * (np.sqrt(x) - np.sqrt(y)) ** 2)

    def _check(self, x):
        if (np.size(self.gamma) > 1 and np.size(self.gamma) != np.size(x)):
            ValueError(
                "'gamma' must be positive scalars or positive ND arrays" +
                " with the same size as 'x'"
            )
