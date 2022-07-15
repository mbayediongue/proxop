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


class Ialpha:
    r"""Compute the proximity operator and the evaluation of gamma*D.

    Where D is the function defined as:


                  /  - ( x* y^{alpha-1})^{1/alpha}   if x >= 0 and y >= 0
        D(x,y) = |
                 \   + inf                             otherwise

   'gamma' is the scale factor

    When the inputs are arrays, the outputs are computed element-wise

    ========
     INPUTS
    ========
    x       - scalar or ND array
    y       - ND array with the same size as 'x' otherwise
    alpha   - >1, scalar or ND array [ default: alpha=2]
    gamma   - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    """

    def __init__(
            self,
            alpha: float or np.ndarray = 2,
            gamma: float or np.ndarray = 1
            ):
        if np.any(gamma <= 0):
            raise Exception("'gamma'  must be strictly positive")
        if np.any(alpha <= 1) or np.size(alpha) > 1:
            raise ValueError("'alpha'  must be a scalar greater than 1")
        self.gamma = gamma
        self.alpha = alpha

    def prox(self, x: np.ndarray, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        scale = self.gamma
        self._check(y)
        alpha = self.alpha
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
        mask = np.logical_or(
            alpha * x >= scale,
            1 - alpha * y / (scale * (alpha - 1))
            < scale / (scale - alpha * x) ** (1 / (alpha - 1)),
        )
        xx = x[mask]
        yy = y[mask]

        if np.size(scale) > 1:
            scale = scale[mask]
        # newton's method
        a = alpha * xx / scale - 1
        b = alpha - 1 - alpha * yy / scale

        def fun_phi(t):
            return t ** (2 * alpha) + a * t ** (alpha + 1) + b * t - alpha + 1

        def der_fun_phi(t):
            return 2 * alpha * t ** (2 * alpha - 1) + alpha * a * t**alpha + b

        # root finding
        eps = 1e-10
        low = np.maximum(0, 1 - alpha * xx / scale) ** (1 / (alpha - 1)) + eps
        root = 50 / alpha + low
        root = newton_(fun_phi, fp=der_fun_phi, x0=root, low=low, high=np.inf)

        # special cases
        root[(a == 0) * (b == 0)] = 1

        # 1st branch
        prox_p[mask] = xx + scale / alpha * (root ** (alpha - 1) - 1)
        prox_q[mask] = yy + scale * (1 - 1 / alpha) * (1 / root - 1)
        return [prox_p, prox_q]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if np.any(x < 0) or np.any(y < 0):
            return np.inf
        alpha = self.alpha
        return np.sum(self.gamma * (-(x**alpha) * y ** (1 - 1 / alpha)))

    def _check(self, x):
        if (np.size(self.gamma) > 1 and np.size(self.gamma) != np.size(x)):
            ValueError(
                "'gamma' must be positive scalars or positive ND arrays" +
                " with the same size as 'x'"
            )
