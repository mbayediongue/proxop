"""
Version : 1.0 ( 06-08-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class ElliotActi:
    r"""Compute the proximity operator and the evaluation of f.

    Where f is the Elliot Activation function:

                   / -|x|-ln(1-|x|)-(x^2)/2  if |x| < 1
           f(x) = |
                  \ +Inf                     otherwise


    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>ElliotActi(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.So, the command >>>ElliotActi.prox(x)
    will return an array with the same shape as 'x'

     INPUTS
    ========
     x     - scalar or ND array

    ========
    Examples
    ========

    Evaluate the 'direct' function :
    >>> ElliotActi()(-0.5)
    0.06814718055994529
    >>>ElliotActi()([-0.2, -0.5, 0, 0.25, 0.3])
    0.08939774826466834

    Compute the proximity operator at a given point :
    >>> ElliotActi().prox(3)
    0.75
    >>> ElliotActi().prox([ -3., 0.5, 6.])
    array([-0.75      ,  0.33333333,  0.85714286])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        return x / (1 + np.abs(x))

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        prox_x = np.zeros_like(x)
        mask = np.abs(x) < 1
        prox_x[mask] = (-np.abs(x[mask]) - np.log(1 - np.abs(x[mask]))
                        - 0.5 * x[mask] ** 2)
        prox_x[np.abs(x) >= 1] = np.inf
        return np.sum(prox_x)
