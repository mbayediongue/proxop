"""
Version : 1.0 (06-18-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class L0Norm:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

      where the function f is defined as:

                                 / 0   if x=0
               f(x) = ||x||_0 = |
                                \  1   otherwise

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    tau   - positive, scalar or ND array with the same size as 'x' [default: tau=1]

    Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>L0Norm(gamma=1)(x) will return
    a scalar even if x is a vector:
    >>> L0Norm(gamma=1)(np.array([-1.,2.,3]))
    3

    But as expected, >>>L0Norm(gamma=1).prox(x) will
    return a vector with the same size as x:

    >>> L0Norm(gamma=1).prox(np.array([-1.,2.,3]))
    array([-0.,  2.,  3.])

    """

    def __init__(
            self,
            gamma: float or np.ndarray = 1.0
    ):
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its elements"
                + " if it is an array) must be strictly positive"
            )
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        gamma = self.gamma
        mask = x**2 >= 2 * gamma
        return x * mask

    def __call__(self, x: np.ndarray) -> float:
        TOL = 1e-10
        return np.sum(self.gamma * (np.abs(x) > TOL))

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise Exception("'gamma' must be either scalar or the same size as 'x'")
