"""
Version : 1.0 ( 06-22-2022).

DEPENDENCIES:
    -'newton.py' - located in the folder 'utils'
    -'Renyi.py'  - located in the folder 'multi'
    
Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""


import numpy as np
from proxop.multi.Renyi import Renyi


class ChiSquare:
    r"""Compute the proximity operator and the evaluation of gamma*D.

    Where D is the  function defined as:


                  /  (x - y)^2 / y   if x >= 0 and y > 0
        D(x,y) = |   0               if x=y=0
                 \  + inf            otherwise

    'gamma' is the scale factor

    When the inputs are arrays, the outputs are computed element-wise

    INPUTS
    ========
    x        - scalar or ND array
    y        - scalar if 'x' is a scalar , ND array with the same size as 'x' otherwise
    gamma    - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.any(gamma <= 0):
            raise ValueError("'gamma'  must be strictly positive")
        self.gamma = gamma

    def prox(self, x: np.ndarray, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(x)
        scale = self.gamma
        return Renyi(alpha=2, gamma=scale).prox(x + 2 * scale, y - scale)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if np.size(x) != np.size(y):
            raise ValueError("'x' and 'y' must have the same size")
        if np.any(x < 0) or np.any(y <= 0):
            return np.inf
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
            y = np.reshape(y, (-1))
        res = (x - y) ** 2 / y
        res[(x == 0) * (y == 0)] = 0
        return np.sum(self.gamma * res)

    def _check(self, x):
        if (np.size(self.gamma) > 1 and np.size(self.gamma) != np.size(x)):
            ValueError(
                "'gamma' must be positive scalars or positive ND arrays" +
                " with the same size as 'x'"
            )
