"""
Version : 1.0 (06-18-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class Truncated:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

      where the function f is defined as:

                                       / x/(1+2*gamma)     if x^2 < w*(1+2*gamma)
               f(x) = min{ x^2, w}  = |
                                      \  1                  otherwise

    'gamma * tau' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    tau   - positive, scalar or ND array with the same size as 'x' [default: tau=1]
    w     - positive, scalar or ND array with the same size as 'x'

    Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>Truncated(w=2, gamma=1)(x) will
    return a scalar even if x is a vector:

    >>> Truncated(w=2, gamma=1)(np.array([-1, 2, 3]))
    5

    But as expected, >>>Truncated(w=2, gamma=1).prox(x)
    will return a vector with the same size as x:

    >>> Truncated(w=2, gamma=1).prox(np.array([-1, 2, 3]))
    array([-0.33333333,  0.66666667,  3.        ])
    """

    def __init__(
            self,
            w: float or np.ndarray,
            gamma: float or np.ndarray = 1.0
    ):

        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it"
                + " is an array) must be strictly positive"
            )
        if np.any(w <= 0):
            raise ValueError(
                "'w' (or all of its components if it"
                + " is an array) must be strictly positive"
            )
        self.gamma = gamma
        self.w = w

    def prox(self, x: np.ndarray) -> np.ndarray:
        gamma = self.gamma
        mask = x**2 < self.w * (1 + 2 * gamma)
        return x / (1 + 2 * gamma * mask)

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.gamma * np.minimum(x**2, self.w))
