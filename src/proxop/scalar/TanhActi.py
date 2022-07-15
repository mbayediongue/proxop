"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class TanhActi:
    r"""Computes the proximity operator and the evaluation of f.

    Where f is the 'Hyperbolic tangent activation' function defined as:

                  / x*arctan(x)+(log(1-x^2)-x^2)/2   if |x| < 1
          f(x) = |
                 \ +Inf                              otherwise


    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function, the output is a scalar (sum of the element-wise
    results).

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.

     INPUTS
    ========
     x     - scalar or ND array

    ========
    Examples
    ========

     Evaluate the function  f:

     >>> TanhActi()( 0.5 )
     0.00581203594

      Compute the result element-wise for vector inputs :

     >>> TanhActi()([ 0.3, 0.25, 0.5] )
     0.0068465198684131146

     Compute the proximity operator at a given point :

     >>> TanhActi().prox(  [-2, .3, 4 ])
     array([-0.96402758,  0.29131261,  0.9993293 ])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.isscalar(x):
            x = np.reshape(x, (-1))
        prox_x = np.zeros(np.shape(x))
        mask = np.abs(x) < 1
        prox_x[mask] = x[mask] * np.arctanh(x[mask]) + 0.5 * (
            np.log(1 - x[mask] ** 2) - x[mask] ** 2
        )
        prox_x[np.abs(x) >= 1] = np.inf
        return np.sum(prox_x)
