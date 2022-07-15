"""
Version : 1.0 (06-18-2022).

DEPENDENCIES:
     'newton.py' in the folder 'utils'

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class EntropyLogSum:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

    Where the function f is defined as:

                       /   xlog(x)+ w*log(delta+x)   if x>0
               f(x) = |    w*log(delta)               if x=0
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
    is computed element-wise SUM. So the command >>>EntropyLogSum(delta=0.01, w=2)(x)
    will return a scalar even if x is a vector:

    >>> EntropyLogSum(delta=0.01, w=2)(np.array([10, 20, 30]))
    202.37911265363505

    But as expected, >>>EntropyLogSum(delta=0.01, w=2).prox(x) $
    will return a vector with the same size as x:

    >>> EntropyLogSum(delta=0.01, w=2).prox(np.array([10, 20, 30]))
    array([ 6.79039076, 16.09718727, 25.67655976])
    """

    def __init__(
        self,
        w: float or np.ndarray = 1.0,
        delta: float or np.ndarray = 1e-10,
        gamma: float or np.ndarray = 1.0
    ):

        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it"
                + " is an array) must be strictly positive"
            )
        if np.any(delta <= 0):
            raise ValueError(
                "'delta' (or all of its components if it"
                + " is an array) must be strictly positive"
            )
        if np.any(w <= 0):
            raise ValueError(
                "'w' (or all of its components if it"
                + " is an array) must be strictly positive"
            )
        self.gamma = gamma
        self.delta = delta
        self.w = w

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check_size(x)
        gamma = self.gamma
        delta = self.delta
        w = self.w

        def f(t):
            return (
                t**2
                + (delta - x + gamma) * t
                + gamma * (delta + t) * np.log(t)
                + delta * (gamma - x)
                + w * gamma
            )

        def df(t):
            return 2 * t + (delta - x + gamma) + gamma * (np.log(t) + delta / t + 1)

        abs_x = np.abs(x)
        prox_x = np.maximum(1, abs_x)
        prox_x = newton_(f, df, prox_x, low=0, high=np.inf)
        return prox_x

    def __call__(self, x):
        self._check_size(x)
        if np.any(x < 0):
            return np.inf

        res = np.zeros_like(1.0 * x)
        mask = x > 0
        w = self.w
        delta = self.delta
        if np.size(delta) > 1:
            delta = delta[mask]
        if np.size(w) > 1:
            w = w[mask]

        res = x[mask] * np.log(x[mask]) + w * np.log(delta + x[mask])
        mask = x == 0
        if np.size(self.delta) > 1:
            delta = self.delta[mask]
        if np.size(self.w) > 1:
            w = self.w[mask]
        res[mask] = w * np.log(delta)
        return np.sum(self.gamma * res)

    def fun(self, x):
        res = np.inf * np.ones_like(1.0 * x)
        mask = x > 0
        w = self.w
        delta = self.delta
        if np.size(delta) > 1:
            delta = delta[mask]
        if np.size(w) > 1:
            w = w[mask]

        res[mask] = x[mask] * np.log(x[mask]) + w * np.log(delta + x[mask])

        mask = x == 0
        if np.size(self.delta) > 1:
            delta = self.delta[mask]
        if np.size(self.w) > 1:
            w = self.w[mask]
        res[mask] = w * np.log(delta)
        return self.gamma * res

    def _check_size(self, x):
        sz_x = np.size(x)
        if (np.size(self.w) > 1) and (np.size(self.w) != sz_x):
            raise ValueError("'w' must be either scalar or the same size as 'x'")
        if (np.size(self.delta) > 1) and (np.size(self.delta) != sz_x):
            raise ValueError("'delta' must be either scalar or the same size as 'x'")
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != sz_x):
            raise ValueError("'gamma' must be either scalar or the same size as 'x'")
