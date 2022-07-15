"""
Version : 1.0 ( 06-15-2022).

DEPENDENCIES:
     - 'newton.py' located in the folder 'utils'

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class EpiPower:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:

                phi(y)= w * |y|^ q

    When the inputs are arrays, the outputs are computed element-wise
     INPUTS
    ========
     y    - ND array
     ksi  - ND array with the same size as 'y'
     w    - positive, scalar or ND array with the same size as ksi [default: w=1.0]
     q    - positive scalar [default: q=2]
    """

    def __init__(
            self,
            ksi: np.ndarray,
            q: float = 2,
            w: float or np.ndarray = 1.0):

        if np.any(w <= 0):
            raise ValueError("'w' must be positive")
        if np.size(q) > 1 or q <= 0:
            raise ValueError("'q' (or must be a positive scalar")
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or have the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.reshape(ksi, (-1))

        self.q = q
        self.w = w
        self.ksi = ksi

    # proximal operator (i.e. the projection on the constraint set)
    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        w = self.w
        q = self.q
        if np.size(y) <= 1:
            y = np.reshape(y, (-1))

        # 2nd branch
        def polynom_phi(t):
            return (
                q * w**2 * t ** (2 * q - 1)
                - q * w * ksi * t ** (q - 1)
                + t
                - np.abs(y)
            )

        def der_phi(t):
            return (
                (2 * q - 1) * q * w**2 * t ** (2 * q - 2)
                - (q - 1) * q * w * ksi * t ** (q - 2)
                + 1
            )

        # starting point
        low = (np.maximum(ksi, 0) / w) ** (1 / q)
        root = 3 * low

        # Finding the root of the polynom with the Newton method
        root = newton_(polynom_phi, fp=der_phi, x0=root, low=low, high=np.inf)
        prox_t = w * root ** q
        prox_y = np.sign(y) * root

        # 1st branch
        mask = w * np.abs(y) ** q <= ksi
        prox_y[mask] = y[mask]
        prox_t[mask] = ksi[mask]
        return [prox_y, prox_t]

    # indicator of the constraint set
    def __call__(self, y: np.ndarray) -> float:
        TOL = 1e-10
        if np.all(self.w * np.abs(y) <= self.ksi) + TOL:
            return 0
        return np.inf

    def _check(self, y):
        if np.size(y) != np.size(self.ksi):
            raise ValueError(" 'y' must have the same size as 'ksi'")
