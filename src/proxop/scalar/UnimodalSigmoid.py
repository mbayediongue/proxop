"""
Original Version : 1.0 (01-02-2019) (updated  06-09-2022).

Author: Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class UnimodalSigmoid:
    r"""Computes the proximity operator and the evaluation of f.

    Where f is the 'unimodal 'sigmoid activation' function defined as

                   / (x+1/2)*log(x+1/2)+(1/2 -x)*log(1/2 -x)-(x^2+1/4)/2  if |x| < 1/2
           f(x) = | -1/4                                                |x| = 1/2
                  \ +Inf                                                otherwise


    When the input 'x' is an array, the output is computed element-wise :

    - for the function, the output is a scalar ( element-wise sum) .

    - But for the proximity operator (method 'prox'), the output has the same shape
    as the input 'x'.

     INPUTS
    ========
     x     - scalar or ND array

    ========
    Examples
    ========

     Evaluate the function  f:

     >>> UnimodalSigmoid()( 0.5 )
     -0.25

      Compute the result element-wise for vector inputs :

     >>> UnimodalSigmoid()([ 0.3, .25, .5] )
     -1.638987568156996

     Compute the proximity operator at a given point :

     >>> UnimodalSigmoid().prox(  [-2, .3, 4 ])
     array([-0.38079708,  0.07444252,  0.48201379])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        return 1 / (1 + np.exp(-x)) - 0.5

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        prox_x = np.zeros_like(1.0 * x)
        mask = np.abs(x) < 0.5
        x_m = x[mask]
        prox_x[mask] = (
            (x_m + 0.5) * np.log(x_m + 0.5)
            + (0.5 - x_m) * np.log(0.5 - x_m)
            - 0.5 * (x_m**2 + 0.25)
        )
        prox_x[np.abs(x) == 0.5] = -0.25
        prox_x[np.abs(x) > 0.5] = np.inf
        return np.sum(prox_x)
