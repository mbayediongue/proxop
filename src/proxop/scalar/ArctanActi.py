"""
Original Version : 1.0 updated  (06-09-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class ArctanActi:
    r"""Compute the proximity operator and the evaluation of f.

     where f is the arc tangent function defined as:

                 / -(2/pi)*log(cos(pi*x/2))-(x^2)/2   if |x| < 1
         f(x) = |
                \ +Inf                 otherwise

    When the input 'x' is an array, the output is computed element-wise :

    - When calling the function (not the proximity operator) the result
    is computed element-wise SUM. So the command >>>ArctanActi()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the
    same shape as the input 'x'. So, the command >>>ArctanActi().prox(x)
    will return an array with the same shape as 'x'


     INPUTS
    ========
     x     - scalar or ND array

    ========
    Examples
    ========

    Evaluate the 'direct' function :
    >>> ArctanActi()(-0.5)
    0.09563560015265157
    >>>ArctanActi()([-0.2, -0.5, 0, 0.25, 0.3])
    0.155203962389311

    Compute the proximity operator at a given point:
    >>> ArctanActi().prox( 3)
    0.7951672353008666
    >>> ArctanActi().prox([ -3., 0.5, 6.])
    array([-0.79516724,  0.29516724,  0.89486309])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        return 2 / np.pi * np.arctan(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        fun_x = np.zeros(np.size(x))
        mask = abs(x) < 1
        fun_x[mask] = -(
            (2 / np.pi) * np.log(np.cos(np.pi * x[mask] / 2)) - x[mask] ** 2 / 2
        )
        fun_x[np.abs(x) >= 1] = np.inf
        return np.sum(fun_x)
