"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class ISRLU:
    r"""Compute the proximity operator of the Inverse Square Root Linear Unit function.

                   / 0                     if x >= 0
           f(x) = | 1-(x^2)/2-sqrt(1-x^2)   if -1<= x < 0
                  \ +Inf                    otherwise


    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>ISRLU()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'. So, the command >>>ISRLU().prox(x)   will return an
    array with the same shape as 'x'.

     INPUTS
    ========
     x     - scalar or ND array

     =======
     Examples
     ========

     >>>ISRLU()( -3 )
     inf
     >>> ISRLU()( [1, -0.25, -0.5] )
     0.009478759663707148

     Compute the proximity operator at a given point :

     >>>ISRLU().prox(-3)
     3.0
     >>>ISRLU(2).prox(  [-2, 3, -0.5] )
     array([-0.89442719,  3.        , -0.4472136 ])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        return np.maximum(x, x / np.sqrt(1 + x**2))

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        result = np.zeros(np.shape(x))
        mask = np.logical_and(x >= -1, x <= 0)
        result[mask] = 1 - x[mask] ** 2 / 2 - np.sqrt(1 - x[mask] ** 2)
        result[x < -1] = np.inf
        return np.sum(result)
