"""
Version : 1.0 (06-18-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class LogSum:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

      where the function f is defined as:


               f(x) = log( delta + |x|)


    'gamma' is the gamma factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    delta -positive, scalar or ND array with the same size as 'x'[default: delata=1e-10]

    Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>LogSum(delta=0.01, gamma=1)(x) will
    return a scalar even if x is a vector:
    >>> LogSum(delta=0.01, gamma=1)(np.array([-1, 2, 3]))
    1.8100251316849367

    But as expected, >>>LogSum(delta=2, gamma=1).prox(x)
    will return a vector with the same size as x:
    >>> LogSum(delta=0.01, gamma=1).prox(np.array([-1, 2, 3]))
    array([0.        , 1.09512492, 2.6197333 ])
    """

    def __init__(
        self,
        delta: float or np.ndarray = 1e-10,
        gamma: float or np.ndarray = 1.0
    ):

        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components"
                + " if it is an array) must be strictly positive"
            )
        if np.any(delta <= 0):
            raise ValueError(
                "'delta' (or all of its components"
                + " if it is an array) must be strictly positive"
            )
        self.gamma = gamma
        self.delta = delta

    def prox(self, x: np.ndarray) -> np.ndarray:
        gamma = self.gamma
        delta = self.delta

        # 1st branch
        prox_x = np.zeros_like(np.shape(x))

        # 2ND branch
        mask = np.abs(x) > np.sqrt(4 * gamma) - delta
        abs_x = np.abs(x[mask])
        if np.size(delta) > 1:
            delta = delta[mask]
        if np.size(gamma) > 1:
            gamma = gamma[mask]

        prox_x[mask] = (
            np.sign(x[mask])
            * 0.5
            * (abs_x - delta + np.sqrt((abs_x + delta) ** 2 - 4 * gamma))
        )
        return prox_x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sum(self.gamma * np.log(self.delta + np.abs(x)))
