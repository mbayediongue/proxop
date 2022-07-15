"""
Version : 1.0 ( 06-16-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class EpiSupport:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:


                               /  * a * x   if x < 0
     phi(y)= sigma_[a,b](y) = |   0         if x = 0
                              \   b * x     otherwise


    When the inputs are arrays, the outputs are computed element-wise
     INPUTS
    ========
     y    - ND array
     kxi  - ND array with the same size as 'y'
     a    - negative, scalar or ND array with the same size as x
     b    - positive, scalar or ND array with the same size as x
    """

    def __init__(
        self,
        ksi: np.ndarray,
        a: float or np.array = -1.0,
        b: float or np.array = 1.0
    ):
        if np.any(a > 0):
            raise ValueError("'a' (or all of its components ) must be negative")
        if np.any(b < 0):
            raise ValueError("'b' (or all of its components ) must be positive")
        if np.size(a) > 1 and (np.size(a) != np.size(ksi)):
            raise ValueError(" 'a' must be a scalar or have the same size as 'ksi'")
        if np.size(b) > 1 and (np.size(b) != np.size(ksi)):
            raise ValueError(" 'b' must be a scalar or have the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.reshape(ksi, (-1))
        self.a = a
        self.b = b
        self.ksi = ksi

    # proximal operator (i.e. the projection on the constraint set)
    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        a = self.a
        b = self.b
        if np.size(y) <= 1:
            y = np.reshape(y, (-1))

        # 4th branch
        prox_p = np.zeros(np.size(y))
        prox_t = np.zeros(np.size(y))

        # 3rd branch
        mask = np.logical_and(a * y > ksi, -y / a <= ksi)
        pp = (y + a * ksi) / (1 + a**2)
        tt = a * pp
        prox_p[mask] = pp[mask]
        prox_t[mask] = tt[mask]

        # 2nd branch
        mask = np.logical_and(b * y > ksi, -y / b <= ksi)
        pp = (y + b * ksi) / (1 + b**2)
        tt = b * pp
        prox_p[mask] = pp[mask]
        prox_t[mask] = tt[mask]

        # 1st branch
        mask = np.logical_and(a * y <= ksi, b * y <= ksi)
        prox_p[mask] = y[mask]
        prox_t[mask] = ksi[mask]
        return [prox_p, prox_t]

    # indicator of the constraint set
    def __call__(self, y) -> float:
        if np.all(self.a * y <= self.ksi) and np.all(self.b * y <= self.ksi):
            return 0
        return np.inf

    def _check(self, y):
        if np.size(y) != np.size(self.ksi):
            raise ValueError(" 'y' must have the same size as 'ksi'")
