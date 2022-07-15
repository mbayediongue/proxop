"""
Version : 1.0 (06-18-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class CapL1:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

    Where f is he capped-L1 (or truncated L1) penalty defined as:

               f(x) = min( 1, theta*|x| )

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    theta  - positive, scalar or ND array with the same size as 'x'[default: theta=1]

     Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>CapL1(theta=0.1)(x) will
    return a scalar even if x is a vector:

    >>>  CapL1(theta=0.1)(np.array([0, 2, -3]))
    0.5

    But as expected, >>>CapL1(theta=0.1).prox(x)
    will return a vector with the same size as x:

    >>> CapL1(theta=0.1).prox(np.array([0, 2, -3]))
    array([ 0. ,  1.9, -2.9])
    """

    def __init__(
            self,
            theta: np.ndarray or float = 1.0,
            gamma: np.ndarray or float = 1.0
    ):

        if np.any(gamma <= 0):
            raise Exception("'gamma'(or all of its elements) must be strictly positive")
        if np.any(theta <= 0):
            raise Exception("'theta'(or all of its elements) must be strictly positive")

        self.gamma = gamma
        self.theta = theta

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        gamma = self.gamma
        theta = self.theta

        p1 = (np.abs(x) < (1 / theta + gamma * theta / 2)) * (gamma * theta**2 < 2)
        p1 = np.sign(x) * (np.abs(x) - gamma * theta) * (np.abs(x) > gamma * theta) * p1
        p2 = (
            x
            * (gamma * theta**2 < 2)
            * (np.abs(x) >= (1 / theta + gamma * theta / 2))
        )
        p3 = x * (np.abs(x) > np.sqrt(2 * gamma)) * (gamma * theta**2 >= 2)

        prox_ = p1 + p2 + p3
        return prox_

    def __call__(self, x: np.ndarray) -> float:
        fun_x = np.minimum(1, self.theta * np.abs(x))
        return np.sum(self.gamma * fun_x)

    def _check(self, x):
        sz_x = np.size(x)
        if (np.size(self.theta) > 1) and (np.size(self.theta) != sz_x):
            raise Exception("'w' must be either scalar or the same size as 'x'")
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != sz_x):
            raise Exception("'gamma' must be either scalar or the same size as 'x'")
