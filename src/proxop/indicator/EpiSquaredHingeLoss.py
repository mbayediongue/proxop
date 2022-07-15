"""
Version : 1.0 ( 06-16-2022).

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


class EpiSquaredHingeLoss:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:
                phi(y)= w * max( y, 0) ^ 2

    When the inputs are arrays, the outputs are computed element-wise
     INPUTS
    ========
     y    - ND array
     kxi  - ND array with the same size as 'y'
     w    - positive, scalar or ND array with the same size as 'ksi'
    """

    def __init__(
            self,
            ksi: np.ndarray,
            w: float or np.ndarray = 1.0
    ):
        if np.any(w <= 0):
            raise ValueError(
                "'w' (or all of its components if it is an array) must be positive"
            )
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or must have" +
                             " the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.reshape(ksi, (-1))
        self.w = w
        self.ksi = ksi

    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        w = self.w

        if np.size(y) <= 1:
            y = np.reshape(y, (-1))

        # Note: same computation as in 'EpiPower.py' for q=2
        def fun_phi(t):
            return 2 * w**2 * t**3 + (1 - 2 * w * ksi) * t - np.abs(y)

        def der_phi(t):
            return 6 * w**2 * t**2 - 2 * w * ksi + 1

        # starting point
        low = (np.maximum(ksi, 0) / w) ** (1 / 2)
        root_init = 3 * low

        # Finding the root of the polynom with the Newton method
        root = newton_(fun_phi, fp=der_phi, x0=root_init, low=low, high=np.inf)
        prox_t = w * (root) ** 2
        prox_p = np.sign(y) * root

        # 1st and 2nd branches
        mask = np.logical_or(w * np.maximum(y, 0) <= ksi, y <= 0)
        prox_p[mask] = y[mask]
        prox_t[mask] = np.maximum(ksi[mask], 0)
        return [prox_p, prox_t]

    def __call__(self, y: np.ndarray) -> float:
        """Indicate if the input 'x' is in the constraint set or not.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        0      if the input 'x' is in the set
        +inf   otherwise
        """

        if np.all(self.w * np.maximum(y, 0) ** 2 <= self.ksi):
            return 0
        return np.inf

    def _check(self, y):
        if np.size(y) != np.size(self.ksi):
            raise ValueError("'y' must have the same size as 'ksi'")
