"""
Version : 1.0 (06-18-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class MCP:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

    Where f is the Minimax Concave Penalty (MCP) defined as:


                  |  lamb^2*beta/2                    if |x| > beta*lamb
        f(x) =   |
                 |  lamb*|x| - x^2/(2*beta)            if |x| <= a*lamb

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    beta  - positive, scalar or ND array with the same size as 'x'
    lamb  - positive, scalar or ND array with the same size as 'x'


     Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>MCP(lamb=2)(x) will
    return a scalar even if x is a vector:

    >>> MCP(lamb=2)(np.array([-1, 2, 3]))
    5.5

    But as expected, >>>MCP(lamb=2).prox(x)
    will return a vector with the same size as x:

    >>> MCP(lamb=2).prox(np.array([-1, 2, 3]))
    array([0, 0, 3]
    """

    def __init__(
        self,
        lamb: float or np.ndarray,
        beta: float or np.ndarray,
        gamma: float or np.ndarray = 1.0
    ):

        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it"
                + " is an array) must be strictly positive"
            )
        if np.any(lamb <= 0):
            raise ValueError(
                "'lamb' (or all of its components if it "
                + "is an array) must be strictly positive"
            )

        if np.any(beta <= 0):
            raise ValueError(
                "'beta' (or all of its components if it"
                + " is an array) must be strictly positive"
            )

        self.lamb = lamb
        self.gamma = gamma
        self.beta = beta

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check_size(x)
        gamma = self.gamma
        beta = self.beta
        lamb = self.lamb
        abs_x = np.abs(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        mask1 = np.abs(x) > np.sqrt(gamma * beta) * lamb
        prox_x = x * mask1
        if np.size(beta - gamma) <= 1 and (beta <= gamma):
            return np.reshape(prox_x, np.shape(x))

        mask = beta > gamma
        if np.size(lamb) > 1:
            lamb = lamb[mask]
        if np.size(beta) > 1:
            beta = beta[mask]
        if np.size(gamma) > 1:
            gamma = gamma[mask]
        filter1 = beta / (beta - gamma) * np.maximum(abs_x[mask] - lamb * gamma, 0)
        prox_x[mask] = np.sign(x[mask]) * np.minimum(filter1, abs_x[mask])

        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        self._check_size(x)
        beta = self.beta
        lamb = self.lamb
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        # 1st branch
        abs_x = np.abs(1.0 * x)
        fun_x = lamb * abs_x - 0.5 * x**2 / beta

        # 2nd branch
        mask = abs_x > beta * lamb
        if np.size(lamb) > 1:
            lamb = lamb[mask]
        if np.size(beta) > 1:
            beta = beta[mask]
        fun_x[mask] = 0.5 * beta * lamb**2

        return np.sum(self.gamma * fun_x)

    def _check_size(self, x):
        sz_x = np.size(x)
        if (np.size(self.lamb) > 1) and (np.size(self.lamb) != sz_x):
            raise ValueError("'lamb' must be either scalar or the same size as 'x'")
        if (np.size(self.beta) > 1) and (np.size(self.beta) != sz_x):
            raise ValueError("'beta' must be either scalar or the same size as 'x'")
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != sz_x):
            raise ValueError("'gamma' must be either scalar or the same size as 'x'")
