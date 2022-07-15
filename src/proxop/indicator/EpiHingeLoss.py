"""
Version : 1.0 ( 06-15-2022).

Authors  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class EpiHingeLoss:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:
                phi(y)= w * max( y, 0)

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
            w: float or np.array = 1.0
    ):

        if np.any(w <= 0):
            raise ValueError("'w' must be positive")
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or have the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.reshape(ksi, (-1))
        self.w = w
        self.ksi = ksi

    # proximal operator (i.e. the projection on the constraint set)
    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        w = self.w

        if np.size(y) <= 1:
            y = np.reshape(y, (-1))

        # 3rd branch
        prox_y = np.maximum(0, y + w * ksi) / (1 + w**2)
        prox_t = w * prox_y

        # 1st and 2nd branches
        mask = np.logical_or(w * np.maximum(y, 0) <= ksi, y <= 0)
        prox_y[mask] = y[mask]
        prox_t[mask] = np.maximum(ksi[mask], 0)
        return [prox_y, prox_t]

    # indicator of the constraint set
    def __call__(self, y: np.ndarray) -> float:
        if np.all(self.w * np.maximum(y, 0) <= self.ksi):
            return 0
        return np.inf

    def _check(self, y):
        if np.size(y) != np.size(self.ksi):
            raise ValueError(" 'y' must have the same size as 'ksi'")
