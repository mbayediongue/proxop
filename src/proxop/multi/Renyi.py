"""
Version : 1.0 ( 06-20-2022).

DEPENDENCIES:
     -'solver_cubic.py' - located in the folder 'utils'
     -'newton.py'  - located in the folder 'utils

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class Renyi:
    r"""Compute the proximity operator and the evaluation of gamma*D.

    Where D is the function defined as:


                  /  (x - y)^alpha / y^{alpha-1}   if x >= 0 and y > 0
        D(x,y) = |   0                             if x=y=0
                 \  + inf                          otherwise

    'gamma' is the scale factor

    When the inputs are arrays, the outputs are computed element-wise.


    INPUTS
    ========
    x          - scalar or ND array
    y          - scalar or ND array with the same size as 'x'
    alpha      - > 1, scalar      [default: alpha=2 ]
    gamma      - scalar or ND array compatible with the blocks of 'y'
                [default: gamma=1]
    """

    def __init__(
            self,
            alpha: float = 2,
            gamma: float or np.ndarray = 1
            ):
        if np.any(gamma <= 0):
            raise ValueError("'gamma'  must be strictly positive")
        if np.any(alpha <= 1) or np.size(alpha) > 1:
            raise ValueError("'alpha'  must be a scalar greter than 1")
        self.gamma = gamma
        self.alpha = alpha

    def prox(self, x: np.ndarray, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        scale = self.gamma
        alpha = self.alpha
        if np.size(x) != np.size(y):
            raise ValueError("'x' and 'y' must have the same size")

        # scalar-like inputs handling
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
            y = np.reshape(y, (-1))

        # 2nd branch
        prox_p = np.zeros(np.shape(x))
        prox_q = np.maximum(0, y)

        # branch selection
        mask = (x > 0) * (
            scale ** (1 / (alpha - 1)) * y
            < (1 - alpha) * (x / alpha) ** (1 + 1 / (alpha - 1))
        )

        xx = x[mask]
        yy = y[mask]

        if np.size(scale) <= 1:
            gg = scale
        else:
            gg = scale[mask]

        # newton's method
        def fun_phi(t):
            return (
                xx / gg * t ** (alpha + 1)
                - yy / gg * t**alpha
                - alpha * t**2
                + 1
                - alpha
            )

        def der_fun_phi(t):
            return (
                (alpha + 1) * xx / gg * t**alpha
                - alpha * yy / gg * t ** (alpha - 1)
                - 2 * alpha * t
            )

        # root finding
        low = (alpha * gg / xx) ** (1 / alpha - 1)
        root = 100 / alpha + low
        root = newton_(fun_phi, fp=der_fun_phi, x0=root, low=low, high=np.inf)

        # 1st branch
        prox_p[mask] = xx - alpha * gg / root ** (alpha - 1)
        prox_q[mask] = yy + (alpha - 1) * alpha / root**alpha
        return [prox_p, prox_q]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if np.any(x < 0) or np.any(y < 0) or np.any((y == 0) * (x != 0)):
            return np.inf

        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
            y = np.reshape(y, (-1))
        alpha = self.alpha
        mask = y == 0
        res = x[mask] ** alpha / y[mask] ** (alpha - 1)
        gg = self.gamma
        if np.size(gg) > 1:
            gg = gg[mask]

        return np.sum(gg * res)
