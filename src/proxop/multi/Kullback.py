"""
Version : 1.0 ( 06-22-2022).

DEPENDENCIES:
    -'newton.py'  - located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class Kullback:
    r"""Compute the proximity operator and the evaluation of gamma*D.

    Where D is the function defined as:


                  /  x * log(x/y)                     if x > 0 and y > 0
        D(x,y) = |   0                                if x=y=0
                 \   + inf                            otherwise

    'gamma' is the scale factor

            When the inputs are arrays, the outputs are computed element-wise
    INPUTS
    ========
    x          - scalar or ND array
    y          - scalar or ND array with the same size as 'x'
    gamma      - scalar or ND array compatible with the blocks of 'y'[default: gamma=1]
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.any(gamma <= 0):
            raise Exception("'gamma'  must be strictly positive")
        self.gamma = gamma

    def prox(self, x: np.ndarray, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        scale = self.gamma

        if np.size(x) != np.size(y):
            raise Exception("'x' and 'y' must have the same size")
        # scalar-like inputs handling
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
            y = np.reshape(y, (-1))

        # 2nd branch
        sz = np.shape(x)
        prox_p = np.zeros(sz)
        prox_q = np.zeros(sz)

        # branch selection
        mask = np.exp(x / scale - 1) >= -y / scale
        xx = x[mask]
        yy = y[mask]
        gg = scale
        if np.size(gg) > 1:
            gg = scale[mask]

        # newton's method
        def fun_phi(t):
            return t * np.log(t) + (xx / gg - 1) * t - 1 / t - yy / gg

        def der_fun_phi(t):
            return np.log(t) + xx / gg + 1 / t**2

        # root finding
        t = np.exp(-xx / gg)
        t = newton_(fun_phi, fp=der_fun_phi, x0=t, low=1 / np.e * t, high=np.inf)

        # 1st branch
        prox_p[mask] = xx + gg * np.log(t) - gg
        prox_q[mask] = yy + gg / t

        return [prox_p, prox_q]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if np.size(x) != np.size(y):
            raise Exception("'x' and 'y' must have the same size")
        if (
            np.any(x < 0)
            or np.any(y < 0)
            or np.any((x == 0) * (y != 0))
            or np.any((y == 0) * (x != 0))
        ):
            return np.inf
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
            y = np.reshape(y, (-1))
        mask = y > 0
        res = x[mask] * np.log(x[mask] / y[mask])
        gg = self.gamma
        if np.size(gg) > 1:
            gg = gg[mask]
        return np.sum(gg * res)

    def _check(self, x):
        if (np.size(self.gamma) > 1 and np.size(self.gamma) != np.size(x)):
            ValueError(
                "'gamma' must be positive scalars or positive ND arrays" +
                " with the same size as 'x'"
            )
