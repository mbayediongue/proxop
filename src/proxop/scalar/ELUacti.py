"""
Version : 1.0 ( 06-08-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class ELUacti:
    r"""Compute the proximity operator and the evaluation of f.

    Where f is the function defined as:

                          /  0                              if  x>=0
                   f(x)= |  (x+w)*ln( (x+w)/w) - x -x^2/2   Iif -w < x < 0
                         \  w - w^2 /2                      if  x=-w
                         \  +INF                            otherwise

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>ELUacti(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same shape
    as the input 'x'.So, the command >>>ELUacti.prox(x)   will return an array
    with the same shape as 'x'.

     INPUTS
    ========
     x     - sclar ND array
     w     - w>=1, scalar or ND array with the same size as 'x'

    =======
    Examples
    ========

    Evaluate the function f:

     >>> ELUacti(w=4)( -3 )
     -2.886294361119891
     >>> ELUacti(2)( [1, -1.5, 3] )
     -0.318147180559945

     Compute the proximity operator at a given point :

     >>> ELUacti(2).prox( 3)
     array([3.])
     >>> ELUacti(2).prox(  [-2, 3, 4, np.e] )
     array([-1.72932943,  3.        ,  4.        ,  2.71828183]
    """

    def __init__(self, w: float or np.ndarray):
        if np.size(w) > 1 and (not isinstance(w, np.ndarray)):
            w = np.array(w)
        self.gamma = 1
        if np.any(w < 1):
            raise Exception(
                "'w' (or all of its components if it is an array) must"
                + " be greater or equal than 1"
            )
        self.w = w

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        w = self.w
        prox_x = w * (np.exp(x) - 1)
        mask = x >= 0
        prox_x[mask] = x[mask]
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        w = self.w
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        result = np.zeros(np.shape(x))
        mask = np.logical_and(x > -w, x < 0)
        ww = w
        if np.size(w) > 1:
            ww = w[mask]

        result[mask] = (
            (x[mask] + ww) * np.log((x[mask] + ww) / ww) - x[mask] - x[mask] ** 2 / 2
        )

        mask2 = x == -w
        if np.size(w) > 1:
            ww = w[mask2]
        result[mask2] = ww - ww**2 / 2
        result[x < -w] = np.inf
        return np.sum(self.gamma * result)
