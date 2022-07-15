"""
Version : 1.0 (06-18-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""


import numpy as np


class CEL0:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

    Where f is the continuous Exact L0 (CEL0) penalty  defined as:

    f(x)= lamb - 0.5*ai^2*(|x| - sqrt(2*lamb)/ai)^2*1_{|x| <= sqrt(2*lamb)/ai}


            'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    a     - positive, scalar or ND array with the same size as 'x'
    lamb  - positive, scalar or ND array with the same size as 'x'


     Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>CEL0(lamb=2, a=3)(x) will
    return a scalar even if x is a vector:

    >>> CEL0(lamb=2, a=3)(np.array([-1, 2, 3]))
    6.0

    But as expected, >>>CEL0(lamb=2, a=3).prox(x)
    will return a vector with the same size as x:

    >>> CEL0(lamb=2, a=3).prox(np.array([-1, 2, 3]))
    array([0, 0, 3])
    """

    def __init__(
        self,
        lamb: float or np.ndarray,
        a: float or np.ndarray,
        gamma: float or np.ndarray = 1.0
    ):
        if np.any(gamma <= 0):
            raise Exception("'gamma'(or all of its elements) must be strictly positive")
        if np.any(lamb <= 0):
            raise Exception("'lamb'(or all of its elements) must be strictly positive")
        if np.any(a <= 0):
            raise Exception("'a'(or all of its elements) must be strictly positive ")
        self.lamb = lamb
        self.gamma = gamma
        self.a = a

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        gamma = self.gamma
        a = self.a
        lamb = self.lamb
        abs_x = np.abs(x)
        sign_x = np.sign(x)

        # 1st case:  a**2*gamma >=1
        mask1 = abs_x > np.sqrt(2 * gamma * lamb)
        prox_ = 1.0 * x * mask1

        if np.size(a**2 * gamma) <= 1 and (a**2 * gamma >= 1):
            return np.reshape(prox_, np.shape(x))

        # 2nd case: a**2*gamma < 1
        mask = a**2 * gamma < 1
        if np.size(lamb) > 1:
            lamb = lamb[mask]
        if np.size(a) > 1:
            a = a[mask]
        if np.size(gamma) > 1:
            gamma = gamma[mask]

        p2 = np.maximum(abs_x[mask] - np.sqrt(2 * lamb) * gamma * a, 0) / (
            1 - a**2 * gamma
        )
        prox_[mask] = sign_x[mask] * np.minimum(abs_x[mask], p2)
        return prox_

    def __call__(self, x: np.ndarray) -> float:
        self._check_size(x)
        a = self.a
        lamb = self.lamb
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        xx = np.abs(x) - np.sqrt(2 * lamb) / a
        mask = xx <= 0
        fun_x = lamb - 0.5 * (a * xx) ** 2 * mask
        return np.sum(self.gamma * fun_x)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise Exception("'gamma' must be either scalar or the same size as 'x'")

    def _check_size(self, x):
        sz_x = np.size(x)
        if (np.size(self.lamb) > 1) and (np.size(self.lamb) != sz_x):
            raise Exception("'lamb' must be either scalar or the same size as 'x'")
        if (np.size(self.a) > 1) and (np.size(self.a) != sz_x):
            raise Exception("'a' must be either scalar or the same size as 'x'")
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != sz_x):
            raise Exception("'gamma' must be either scalar or the same size as 'x'")
