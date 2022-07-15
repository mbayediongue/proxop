"""
Version : 1.0 ( 06-22-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from scipy.optimize import minimize


class PersVapnik:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:


                 /   inf_{y \in R}( |y| + 1_{[-eps*xi, eps*ksi]}    if xi >= 0
      f(y,xi) = |
                \    + inf                                         otherwise



    'gamma' is the scale factor


    INPUTS
    ========
    y          -  scalar or ND array
    xi         -  scalar or ND array with the same size as 'y'
    eps        - scalar or ND array with the same size as 'y'
    gamma      - positive, scalar or ND array compatible with the blocks of 'y'
                [default: gamma=1]
    """

    def __init__(
            self,
            eps: float or np.ndarray = 1,
            xi: float or np.ndarray = 1,
            gamma: float or np.ndarray = 1
            ):
        if np.any(gamma <= 0):
            raise Exception("'gamma'  must be strictly positive")

        if np.any(eps <= 0):
            raise Exception("'eps'  must be strictly positive")

        if np.size(eps) != np.size(xi) and np.size(eps) > 1 and np.size(xi) > 1:
            raise Exception(
                "'eps'  and 'xi' must have the same size when they are not scalars"
            )

        if np.size(gamma) <= 1:
            gamma = np.reshape(gamma, (-1))
        if np.size(xi) <= 1:
            xi = np.reshape(xi, (-1))
        if np.size(eps) <= 1:
            eps = np.reshape(eps, (-1))

        self.gamma = gamma
        self.eps = eps
        self.xi = xi

    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        scale = self.gamma
        eps = self.eps
        xi = self.xi

        if np.size(y) <= 1:
            y = np.reshape(y, (-1))
        sz = np.shape(y)
        if np.size(y) > 1:
            if np.size(xi) <= 1:
                xi = xi * np.ones(sz)
            if np.size(eps) <= 1:
                eps = eps * np.ones(sz)
            if np.size(scale) <= 1:
                scale = xi * np.ones(sz)

        abs_y = np.abs(y)
        sign_y = np.sign(y)

        # 1st branch
        prox_p = np.zeros_like(1.0 * y)
        prox_q = np.zeros_like(1.0 * xi)

        # 2nd branch
        mask = (xi <= -scale * eps) * (abs_y > scale)
        prox_p[mask] = y[mask] - scale[mask] * sign_y[mask]

        # 3rd branch
        mask = (xi > -scale * eps) * (abs_y > eps * xi + scale * (1 + eps**2))
        prox_p[mask] = y[mask] - scale[mask] * sign_y[mask]
        prox_q[mask] = eps[mask] + scale[mask] * eps[mask]

        # 4th branch
        mask = (
            (abs_y > -xi / eps)
            * (abs_y > eps * xi + scale * (1 + eps**2))
            * (abs_y <= eps * xi)
        )
        prox_q[mask] = (xi[mask] + eps[mask] * sign_y[mask]) / (1 + eps[mask] ** 2)
        prox_p[mask] = eps[mask] * sign_y[mask] * prox_q[mask]

        return [prox_p, prox_q]

    def __call__(self, y: np.ndarray) -> float:
        eps = self.eps
        xi = self.xi
        xi = np.reshape(xi, (-1))
        eps = np.reshape(eps, (-1))
        y = np.reshape(y, (-1))
        g = np.reshape(self.gamma, (-1))

        def fun_phi(t):
            mask = (y - t >= -eps * xi) * (y - t <= eps * xi)
            return np.sum(np.abs(y) + mask)

        res = minimize(fun_phi, y)

        return np.sum(fun_phi(g * res.x))

    def _multiply_tuple(self, tup):
        res = 1
        for i in tup:
            res = res * i
        return res

    def _check(self, y):
        if np.size(self.xi) > 1 and np.size(self.xi) != np.size(y):
            raise Exception(
                "'xi' must be scalar or an ND array with the same size as 'y'"
            )
        if np.size(self.eps) > 1 and np.size(self.eps) != np.size(y):
            raise Exception(
                "'eps' must be scalar or an ND array with the same size as 'y'"
            )
