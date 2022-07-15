"""
Version : 1.0 ( 06-15-2022).

Author : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class EpiAbsValue:
    r"""Compute the projection onto the epigraph of phi.

    Where phi is the function defined as:

                phi(y)= w * |y|

    When the inputs are arrays, the outputs are computed element-wise

     INPUTS
    ========
     y    - ND array
     ksi  - ND array with the same size as 'y'
     w    - positive, scalar or ND array with the same size as 'ksi' [default: w=1.0]
    """

    def __init__(
            self,
            ksi: np.ndarray,
            w: float or np.ndarray = 1.0
    ):

        if np.any(w <= 0):
            raise ValueError("'w' must be positive")
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or must have" +
                             " the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.reshape(ksi, (-1))
        self.w = w
        self.ksi = ksi

    # proximal operator (i.e. the projection onto the constraint set)
    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        w = self.w
        if np.size(y) <= 1:
            y = np.reshape(y, (-1))

        # 2nd branch
        y_abs = np.abs(y)
        coef = np.maximum(0, y_abs + w * ksi) / (1 + w**2)
        prox_p = np.sign(y) * coef
        prox_t = w * coef

        # 1st branch
        mask = w * y_abs < ksi
        prox_p[mask] = y[mask]
        prox_t[mask] = ksi[mask]
        return [prox_p, prox_t]

    # indicator of the constraint set
    def __call__(self, y: np.ndarray) -> float:
        if np.all(self.w * np.abs(y) <= self.ksi):
            return 0
        return np.inf

    def _check(self, y):
        if np.size(y) != np.size(self.ksi):
            raise ValueError(" 'y' must have the same size as 'ksi'")
