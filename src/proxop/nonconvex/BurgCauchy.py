"""
Version : 1.0 ( 06-18-2022).

Author  : Mbaye DIONGUE

DEPENDENCIES:
   -'newton.py' in the folder 'utils'

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class BurgCauchy:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

      where the function f is defined as:


                       /   -log(x)+ w*log( delta + x^2 )   if x>0
               f(x) = |
                      \   +inf                      otherwise

                with w>0 and delta>0

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x    - ND array
    gamma- positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    delta- positive, scalar or ND array with the same size as 'x'[default: delata=1e-10]
    w    - positive, scalar or ND array with the same size as 'x' [default: w=1]

    Note: When calling the function (not the proximity operator) the result
    is computed element-wise SUM. So the command >>>BurgCauchy(delta=0.01, w=2)(x) will
    return a scalar even if x is a vector:

    >>> BurgCauchy(delta=0.01, w=2)(np.array([10, 20, 30]))
    26.098816

    But as expected, >>>BurgCauchy(delta=0.01, w=2).prox(x)
    will return a vector with the same size as x:

    >>> BurgCauchy(delta=0.01, w=2).prox(np.array([10, 20, 30]))
    array([ 9.69046116, 19.84886296, 29.89966593])
    """

    def __init__(
        self,
        w: np.ndarray or float = 1.0,
        delta: np.ndarray or float = 1e-10,
        gamma: np.ndarray or float = 1.0
    ):
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma'(or all of its elements)" + " must be strictly positive"
            )
        if np.any(delta <= 0):
            raise ValueError(
                "'delta'(or all of its elements)" + " must be strictly positive"
            )
        if np.any(w <= 0):
            raise ValueError(
                "'w' (or all of its elements)" + " must be strictly positive"
            )
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

        def f(t):
            return (
                t**4
                - x * t**3
                + (delta + 2 * gamma * w - gamma) * t**2
                - delta * x * t
                - delta * gamma
            )

        def df(t):
            return (
                4 * t**3
                - 3 * x * t**2
                + 2 * (delta + 2 * gamma * x - gamma) * t
                - delta * x
            )

        abs_x = np.abs(x)
        prox_x = np.maximum(1, abs_x)
        low = 0
        prox_x = newton_(f, df, prox_x, low=low, high=np.inf)
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if np.any(x <= 0):
            return np.inf
        return np.sum(-np.log(x) + self.w * np.log(self.delta + x**2))

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("'gamma' must be either scalar or the same size as 'x'")
