"""
Version : 1.0 (06-10-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class SmoothedFermiDiracEntropy:
    r"""Computes the proximity operator and the evaluation of f.

    Where f is the 'Smoothed Fermi-Dirac Entropy' function defined as:

                      /  xlog(x)+ (1-x)log(1-x) -0.5*x^2        if 0< x < 1
               f(x)= |   0                                      if x in {0, 1}
                     \  +INF                                    otherwise


    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function, the output is a scalar (sum of the
    element-wise results).

    - But for the proximity operator (method 'prox'), the output has the
    same shape as the input 'x'.

     INPUTS
    ========
     x     - scalar or ND array

    =======
    Examples
    ========

     Evaluate the function  f:

     >>> SmoothedFermiDiracEntropy()( 0.5 )
     -0.8181471805599453

      Compute the result element-wise sum for vector inputs :

     >>> SmoothedFermiDiracEntropy()( [1, .3, 0.5] )
     -1.4740114826148387

     Compute the proximity operator at a given point :
     >> SmoothedFermiDiracEntropy().prox(   3 )
     0.9525741268224334
     >>> SmoothedFermiDiracEntropy().prox(  [-2, 3, 4 ])
     array([0.11920292, 0.95257413, 0.98201379])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        return 1 / (1 + np.exp(-x))

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        fun_x = np.zeros_like(1.0 * x)
        mask = np.logical_and(x > 0, x < 1)
        fun_x[mask] = (
            x[mask] * np.log(x[mask])
            + (1 - x[mask]) * np.log(1 - x[mask])
            - 0.5 * x[mask] ** 2
        )
        mask2 = np.logical_or(x < 0, x > 1)
        fun_x[mask2] = np.inf

        return np.sum(fun_x)
