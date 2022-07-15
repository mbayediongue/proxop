"""
Version : 1.0 (06-18-2022).

DEPENDENCIES:
     'solver_cubic.py' in the folder 'utils'

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.solver_cubic import solver_cubic


class BurgLogSum:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

      where the function f is defined as:

                       /   -log(x)+ w*log(delta+x)   if x>0
               f(x) = |
                      \   +inf                      otherwise

    with w>0 and delta>0
    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    delta -positive, scalar or ND array with the same size as 'x'[default: delata=1e-10]
    w     - positive, scalar or ND array with the same size as 'x' [default: w=1]


    Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>BurgLogSum(w=2)(x) will
    return a scalar even if x is a vector:

    >>> BurgLogSum(w=2)(np.array([10, 20, 30]))
    8.699514748246859

    But as expected, >>>BurgLogSum(w=2).prox(x)
    will return a vector with the same size as x:

    >>> BurgLogSum(w=2).prox(np.array([10, 20, 30]))
    array([ 9.89897949, 19.94987437, 29.96662955])
    """

    def __init__(
        self,
        w: np.ndarray or float = 1.0,
        delta: np.ndarray or float = 1e-10,
        gamma: np.ndarray or float = 1.0,
    ):

        if np.any(gamma <= 0):
            raise ValueError("'gamma' (or all of its elements)" +
                             " must be strictly positive")
        if np.any(delta <= 0):
            raise ValueError("'delta' (or all of its elements)" +
                             " must be strictly positive")
        if np.any(w <= 0):
            raise ValueError("'w' (or all of its elements)" +
                             " must be strictly positive")
        self.gamma = gamma
        self.delta = delta
        self.w = w

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        gamma = self.gamma
        delta = self.delta
        w = self.w
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        # Solve the cubic equation
        prox_x, p2, p3 = solver_cubic(
            1, delta - x, w * gamma - delta * x - gamma, -gamma * delta
        )

        def f(t):
            res = np.inf * np.ones_like(t)
            mask_ = t > 0
            w_ = w
            delta_ = delta
            if np.size(delta) > 1:
                delta_ = delta[mask_[0]]
            if np.size(w) > 1:
                w_ = w[mask_[0]]
            res[mask_] = -np.log(t[mask_]) + w_ * np.log(delta_ + t[mask_])
            return 0.5 * np.abs(x - t) ** 2 + gamma * res

        mask = np.logical_and(
            np.isreal(p2) * (np.real(p2) > 0), f(np.real(p2)) < f(prox_x)
        )

        prox_x[mask] = np.real(p2[mask])
        mask = np.logical_and(
            np.isreal(p2) * (np.real(p3) > 0), f(np.real(p3)) < f(prox_x)
        )
        prox_x[mask] = np.real(p3[mask])

        return np.reshape(prox_x, np.shape(x))

    def __call__(self, x: np.ndarray) -> float:
        if np.any(x <= 0):
            return np.inf
        res = -np.log(x) + self.w * np.log(self.delta + x)
        return np.sum(self.gamma * res)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("'gamma' must be either scalar or the same size as 'x'")
