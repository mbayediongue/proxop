"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class ISRU:
    r"""Compute the proximity operator of the Inverse Square Root Unit function.

                   / -(x^2)/2-sqrt(1-x^2)   if |x| <= 1
           f(x) = |
                  \ +Inf                     otherwise


    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>ISRU()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same shape
    as the input 'x'.
    So, the command >>>ISRU().prox(x)   will return an array with the same shape as 'x'

     INPUTS
    ========
     x     - scalar or ND array

     =======
     Examples
     ========

     >>>ISRU()( -3 )
     INF
     >>> ISRU()( [1, -0.25, -0.5] )
     -2.490521240336293

     Compute the proximity operator at a given point :

     >>>ISRU().prox(-3)
     -0.9486832980505138
     >>>ISRU().prox(  [-2, 3, -0.5] )
      array([-0.89442719,  0.9486833 , -0.4472136 ])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        return x / np.sqrt(1 + x**2)

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        result = np.zeros(np.shape(x))
        mask = np.abs(x) <= 1
        result[mask] = -x[mask] ** 2 / 2 - np.sqrt(1 - x[mask] ** 2)
        result[np.abs(x) > 1] = np.inf
        return np.sum(result)
