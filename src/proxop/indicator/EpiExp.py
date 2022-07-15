"""
Version : 1.0 ( 06-16-2022).

DEPENDENCIES:
     - 'newton.py' located in the folder 'utils'

Authors  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class EpiExp:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:
                phi(y)= w * max( y, 0)

    When the inputs are arrays, the outputs are computed element-wise
     INPUTS
    ========
     y    - ND array
     ksi  - ND array with the same size as 'y'
     w    - positive, scalar or ND array with the same size as 'x' [default: w=1.0]
    """

    def __init__(
            self,
            ksi: np.ndarray,
            w: float or np.ndarray = 1.0
    ):

        if np.any(w <= 0):
            raise ValueError("'w' must be positive")
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or" +
                             " must have the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.array([ksi])
        self.w = w
        self.ksi = ksi

    # proximal operator (i.e. the projection on the constraint set)
    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        w = self.w

        if np.size(y) <= 1:
            y = np.array([y])
        # 1st branch
        prox_y = np.copy(y)
        prox_t = np.maximum(ksi, w * np.exp(prox_y))

        eps = 1e-16
        mask = np.logical_and(w * np.exp(y) > ksi, y > np.log(eps) + 1)
        yy = y[mask]
        xx = ksi[mask]
        ww = w
        if np.size(ww) > 1:
            ww = w[mask]

        # Newton's method
        def polynom_phi(t):
            return t**2 - xx * t + np.log(t) * yy

        def der_phi(t):
            return 2 * t - xx + 1 / (t + eps)

        # starting point
        low = np.maximum(eps, xx)
        root = np.abs(xx)

        # Finding the root of the polynom with the Newton method
        root = newton_(polynom_phi, fp=der_phi, x0=root, low=low, high=np.inf)

        # 2nd branch
        prox_y[mask] = np.log(root / ww)
        prox_t[mask] = root

        return [prox_y, prox_t]

    # indicator of the constraint set
    def __call__(self, y: np.ndarray) -> float:
        if np.all(self.w * np.maximum(y, 0) <= self.ksi):
            return 0
        return np.inf

    def _check(self, y):
        if np.size(y) != np.size(self.ksi):
            raise ValueError(" 'y' must have the same size as 'ksi'")
