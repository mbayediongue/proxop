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


class EpiLog:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:

                          / -w*log(y)    if y>0
                phi(y)=  |
                         \ + inf        otherwise

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
            raise ValueError(
                "'w' (or all of its components if it is an array) must be positive"
            )
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or must have" +
                             " the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.array([ksi])
        self.w = w
        self.ksi = ksi

    # proximal operator (i.e. the projection on the constraint set)
    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        w = self.w

        # Newton's method .
        eps = 1e-20

        def fun_phi(t):
            pp = 0.5 * (y + np.sqrt(y**2 + 4 * w * t))
            return -w * np.log(pp + eps) - ksi - t

        def der_phi(t):
            dp = w / np.sqrt(y**2 + 4 * w * t)
            pp = 0.5 * (y + np.sqrt(y**2 + 4 * w * t))
            return -w * dp / (pp + eps) - 1

        # starting point
        low = 0
        root_init = np.abs(y)

        # Finding the root of the polynom with the Newton method
        root = newton_(fun_phi, fp=der_phi, x0=root_init, low=low, high=np.inf)
        prox_p = 0.5 * (y + np.sqrt(y**2 + 4 * w * root))
        prox_t = ksi + root

        return [prox_p, prox_t]

    # indicator of the constraint set
    def __call__(self, y: np.ndarray) -> float:
        if np.all(y > 0) and np.all(-self.w * np.log(y) <= self.ksi):
            return 0
        return np.inf

    def _check(self, y):
        if np.size(y) != np.size(self.ksi):
            raise ValueError(" 'y' must have the same size as 'ksi'")
