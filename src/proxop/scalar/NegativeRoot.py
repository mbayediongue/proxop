"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class NegativeRoot:
    r"""Computes the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:


                  /  - x^(1/q)    if x>=0
           f(x)= |
                 \    +INF                 otherwise

                 with q>=1

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function, the output is a scalar (sum of the element-wise
    results ).

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.

     INPUTS
    ========
     x     - scalar or ND array
     q     - scalar >=1
     gamma - positive, scalar or ND array with the same size as 'x' (default: gamma=1)

    =======
    Examples
    ========

     Evaluate the function  f:

     >>> NegativeRoot(3)( 8 )
     -2.0

      Compute the result element-wise sum for vector input :

     >>> NegativeRoot(q=2, gamma=[1, 2, 3])( [1, 16, 9] )
     -18.0

     Compute the proximity operator at a given point :

     >>> NegativeRoot(2).prox(  [-2, 3, 4 ])
     array([0.49278772, 1.3453765 , 1.43519754])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the
     function 'gamma*f'

     >>> NegativeRoot( 2, gamma=[1,2,3,3]).prox( [-2, 3, 4, np.e ] )
     array([0.49278772, 1.37090672, 1.47180413, 1.36953862])
    """

    def __init__(
            self,
            q: float, gamma: float or np.ndarray = 1
            ):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array)"
                + " must be strictly positive"
            )
        self.gamma = gamma
        if np.any(q < 1) or np.size(q) > 1:
            raise ValueError("'q' must be a scalar greater than 1")
        self.q = q

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)

        self._check(x)
        scale = self.gamma

        abs_x = np.abs(x)
        q = self.q
        if q == 1:
            return np.maximum(0, x + scale)

        def polynom_phi(t):
            return t ** (2 * q - 1) - x * t ** (q - 1) - scale / q

        def der_phi(t):
            return (2 * q - 1) * t ** (2 * q - 2) - (q - 1) * x * t ** (q - 2)

        # starting point
        prox_x = abs_x

        # Finding the root of the polynom with Newton method
        prox_x = newton_(polynom_phi, fp=der_phi, x0=prox_x, low=0, high=np.inf)

        return prox_x ** (1 / q)

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        xx = np.zeros(np.shape(x))
        mask = x >= 0
        xx[mask] = -x[mask] ** (1 / self.q)
        xx[np.logical_not(mask)] = np.inf

        return np.sum(self.gamma * xx)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.shape(self.gamma) != np.shape(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
