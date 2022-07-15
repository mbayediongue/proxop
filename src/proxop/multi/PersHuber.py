"""
Version : 1.0 ( 06-22-2022).

DEPENDENCIES:
     - 'PersSquare.py' located in the folder 'multi'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.multi.PersSquare import PersSquare


class PersHuber:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:


                 /   rho*|y| - 0.5*xi*rho^2     if xi > 0 and |y| > xi*rho
      f(y,xi) = |    | x |^2/(2*xi)             if xi > 0 and |y| <= xi*rho
                |    rho * |y|                  if y=0
                \    + inf                      otherwise



    'gamma' is the scale factor

            When the inputs are arrays, the outputs are computed element-wise
    INPUTS
    ========
    y         - scalar or ND array
    xi        - scalar or ND array compatible with the blocks of 'y'
    rho        - positive, scalar or ND array with the same size as 'y'
    gamma     - positive, scalar or ND array compatible with the blocks of 'x'
                 [default: gamma=1]
    """

    def __init__(self,
                 rho: np.ndarray,
                 xi: float or np.ndarray,
                 gamma: float or np.ndarray,
                 ):
        if np.any(gamma <= 0):
            raise Exception("'gamma'  must be strictly positive")

        if np.any(rho <= 0):
            raise Exception("'rho'  must be strictly positive")

        if np.any(np.size(rho) != np.size(xi)):
            raise Exception("'rho'  and 'xi' must have the same size")

        if np.size(gamma) <= 1:
            gamma = np.array([gamma])
        if np.size(xi) <= 1:
            xi = np.array([xi])
        if np.size(rho) <= 1:
            rho = np.array([rho])

        self.gamma = gamma
        self.xi = xi
        self.rho = rho

    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        scale = self.gamma
        rho = self.rho
        xi = self.xi

        if np.size(y) <= 1:
            y = np.reshape(y, (-1))
        sz0 = np.shape(y)
        sz = np.shape(y)
        if np.size(y) > 1:
            if np.size(xi) <= 1:
                xi = xi * np.ones(sz)
            if np.size(rho) <= 1:
                rho = rho * np.ones(sz)
            if np.size(scale) <= 1:
                scale = xi * np.ones(sz)
        abs_y = np.abs(y)
        sign_y = np.sign(y)
        rho2 = rho**2

        # 4th branch
        y_ = y
        if np.size(xi) > 1:
            y_ = np.reshape(y, np.shape(y) + tuple([1]))
            prox_p, prox_q = PersSquare(xi, 1, len(y_.shape) - 1).prox(y_)
        else:
            prox_p, prox_q = PersSquare(xi[0], 1).prox(y_)
        prox_p = np.reshape(prox_p, sz0)
        prox_q = np.reshape(prox_q, np.shape(xi))

        # 1st branch
        mask = (2 * xi * scale + y**2 <= 0) * (abs_y < scale * rho)
        prox_p[mask] = 0
        prox_q[mask] = 0

        # 2nd branch
        mask = (xi <= -0.5 * scale * rho2) * (abs_y > scale * rho)
        scale_m = scale
        rho_m = rho
        if np.size(scale) > 1:
            scale_m = scale[mask]
        if np.size(rho) > 1:
            rho_m = rho[mask]
        prox_p[mask] = y[mask] - rho_m * scale_m * sign_y[mask]

        # 3rd branch
        mask = (xi > -0.5 * scale * rho2) * (
            abs_y > rho * xi + scale * rho * (1 + 0.5 * rho2)
        )
        scale_m = scale
        rho_m = rho
        xi_m = xi
        if np.size(scale) > 1:
            scale_m = scale[mask]
        if np.size(rho) > 1:
            rho_m = rho[mask]
        if np.size(xi) > 1:
            xi_m = xi[mask]
        prox_p[mask] = y[mask] - rho_m * scale_m * sign_y[mask]
        prox_q[mask] = xi_m + 0.5 * scale_m * rho_m**2

        return [np.reshape(prox_p, np.shape(y)), prox_q]

    def __call__(self, y: np.ndarray) -> float:
        if np.any(y < 0):
            return np.inf

        if np.size(y) <= 1:
            y = np.reshape(y, (-1))

        xi = self.xi
        rho = self.rho

        res = rho * np.abs(y) - 0.5 * xi * rho**2
        mask = np.logical_and(xi > 0, np.abs(y) <= xi * rho)
        xi_m = xi
        if np.size(xi) > 1:
            xi_m = xi[mask]
        res[mask] = 0.5 * np.abs(y[mask]) ** 2 / xi_m

        mask = y == 0
        rho_m = rho
        if np.size(rho) > 1:
            rho_m = rho[mask]
        res[mask] = rho_m * y[mask]

        return np.sum(self.gamma * res)

    def _check(self, y):
        if np.size(self.xi) > 1 and np.size(self.xi) != np.size(y):
            raise Exception(
                "'xi' must be scalar or an ND array with the same size as 'y'"
            )
        if np.size(self.rho) > 1 and np.size(self.rho) != np.size(y):
            raise Exception(
                "'rho' must be scalar or an ND array with the same size as 'y'"
            )
