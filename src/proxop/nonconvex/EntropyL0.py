"""
Version : 1.0 (06-18-2022).

DEPENDENCIES:
     'lambert_W.py' in the folder 'utils'

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.lambert_W import lambert_W


class EntropyL0:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

      where the function f is defined as:

                                       /   x*log(x) + w      if x>0
               f(x) = min{ x^2, w}  = |    0                 if x=0
                                      \    +inf             otherwise

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise SUM.

     INPUTS
    ========
    x     - ND array
    w     - positive, scalar or ND array with the same size as 'x'
    gamma - positive, scalar or ND array with the same size as 'x'[default: gamma=1]


    Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>EntropyL0(w=2)(x) will
    return a scalar even if x is a vector:

    >>> EntropyL0(w=2)(np.array([1, 2, 3]))
    10.68213122712422

    But as expected, >>>EntropyL0(w=2).prox(x)
    will return a vector with the same size as x:

    >>> EntropyL0(w=2).prox(np.array([1, 2, 3]))
    array([0.       , 0.       , 1.5571456])
    """

    def __init__(
            self,
            w: float or np.ndarray,
            gamma: float or np.ndarray = 1.0
    ):
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its elements) must be strictly positive"
            )
        if np.any(w <= 0):
            raise ValueError("'w' (or all of its elements) must be strictly positive")
        self.gamma = gamma
        self.w = w

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        gamma = self.gamma
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        prox_x = np.zeros(np.shape(x))
        # 3rd branch
        # we use a taylor approximation to avoid divergence when the input is too big
        u = np.log(1 / gamma) + x / gamma - 1
        mask = u > 100

        prox_x[mask] = u[mask] - u[mask] / (1 + u[mask]) * np.log(u[mask])

        # we use the Lambert_W function otherwise
        mask = np.logical_not(mask)
        if np.size(gamma) > 1:
            gamma = gamma[mask]
        prox_x[mask] = gamma * lambert_W(1 / gamma * np.exp(x[mask] / gamma - 1))
        gamma = self.gamma
        mask = prox_x**2 + 2 * gamma * prox_x < 2 * self.w * gamma
        # 1st and 2nd branch
        prox_x[mask] = 0
        return prox_x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        if np.any(x < 0):
            return np.inf
        mask = x > 0
        w = self.w
        if np.size(w) > 1:
            w = w[mask]
        g = self.gamma
        if np.size(g) > 1:
            g = g[mask]
        fun_x = g * (x[mask] * np.log(x[mask]) + w)
        return np.sum(fun_x)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("'gamma' must be either scalar or the same size as 'x'")
