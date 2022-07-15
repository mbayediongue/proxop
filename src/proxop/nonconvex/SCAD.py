"""
Version : 1.0 (06-18-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class SCAD:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

    where f is Smoothly Clipped Absolute Deviation (SCAD)  defined as:


                  |  lamb*|x|                               if |x| <= lamb
        f(x) =   |  -(lamb^2-2*a*lamb*|x| + x^2)/[2*(a-1)]  if lamb < |x| <= a*lamb
                 |  (a+1)*lamb^2/2                          if |x| > a*lamb

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    a     - >2, scalar or ND array with the same size as 'x'
    lamb  - positive, scalar or ND array with the same size as 'x'


     Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>SCAD(lamb=2, a=3)(x) will
    return a scalar even if x is a vector:

    >>> SCAD(lamb=2, a=3)(np.array([-1, 2, 3]))
    11.75

    But as expected, >>>SCAD(lamb=2, a=3).prox(x)
    will return a vector with the same size as x:

    >>> SCAD(lamb=2, a=3).prox(np.array([-1, 2, 3]))
    array([0., 0., 1.])
    """

    def __init__(
        self,
        lamb: float or np.ndarray,
        a: float or np.ndarray,
        gamma: float or np.ndarray = 1.0
    ):

        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it"
                + " is an array) must be strictly positive"
            )
        if np.any(lamb <= 0):
            raise Exception(
                "'lamb' (or all of its components if it"
                + " is an array) must be strictly positive"
            )
        if np.any(a <= 2):
            raise Exception(
                "'a' (or all of its components if it"
                + " is an array) must be greater than 2"
            )
        self.lamb = lamb
        self.gamma = gamma
        self.a = a

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check_size(x)
        gamma = self.gamma
        a = self.a
        lamb = self.lamb
        abs_x = np.abs(x)
        sign_x = np.sign(x)

        # 1st case:  a <= 1+gamma
        mask1 = abs_x <= 0.5 * (a + 1 + gamma) * lamb
        mask2 = 1 - mask1
        prox_x = 1.0 * sign_x * np.maximum(abs_x - lamb * gamma, 0) * mask1 + x * mask2

        if np.size(a - gamma) <= 1 and (a <= 1 + gamma):
            return np.reshape(prox_x, np.shape(x))

        # 2nd case:  a>1+gamma
        mask = a > 1 + gamma
        if np.size(lamb) > 1:
            lamb = lamb[mask]
        if np.size(a) > 1:
            a = a[mask]
        if np.size(gamma) > 1:
            gamma = gamma[mask]
        mask1 = abs_x[mask] <= (1 + gamma) * lamb
        mask2 = (abs_x[mask] > (1 + gamma) * lamb) * (abs_x[mask] <= a * lamb)
        mask3 = abs_x[mask] > a * lamb
        x_ = x[mask]
        p1 = (
            1.0 * sign_x[mask] * np.maximum(abs_x[mask] - lamb * gamma, 0) * mask1
            + ((a - 1) * x_ - sign_x[mask] * a * lamb * gamma) / (a - 1 - gamma) * mask2
            + x_ * mask3
        )

        prox_x[mask] = p1
        return np.reshape(prox_x, np.shape(x))

    def __call__(self, x):
        self._check_size(x)
        a = self.a
        lamb = self.lamb
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        # 1st branch
        abs_x = np.abs(1.0 * x)
        fun_x = lamb * abs_x

        # 2nd branch
        mask = np.logical_and(abs_x > lamb, abs_x <= a * lamb)
        fun_x[mask] = -(0.5 * (lamb**2 - 2 * a * lamb * abs_x + x**2) / (a - 1))[
            mask
        ]

        # 3rd branch
        mask = abs_x > a * lamb
        if np.size(lamb) > 1:
            lamb = lamb[mask]
        if np.size(a) > 1:
            a = a[mask]
        fun_x[mask] = 0.5 * (a + 1) * lamb**2

        return np.sum(self.gamma * fun_x)

    def _check_size(self, x):
        sz_x = np.size(x)
        if (np.size(self.lamb) > 1) and (np.size(self.lamb) != sz_x):
            raise Exception("'lamb' must be either scalar or the same size as 'x'")
        if (np.size(self.a) > 1) and (np.size(self.a) != sz_x):
            raise Exception("'a' must be either scalar or the same size as 'x'")
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != sz_x):
            raise Exception("'gamma' must be either scalar or the same size as 'x'")
