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


class LogInverse:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is 'logarithm+inverse' function defined as:

                      / gamma*tau( - log(x) + w * x^(-1) )     if  x> 0
               f(x)= |
                     \  +INF                                  otherwise

        with w > 0 and q >= 1

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>LogInverse()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'. So, the command >>>LogInverse().prox(x)   will return an
    array with the same shape as 'x'.


     INPUTS
    ========
     x     - scalar or ND array
     w     - positive , scalar or ND array with the same size as 'x'
     gamma - positive, scalar or ND array with the same size as 'x' (default: gamma=1)

    =======
    Examples
    ========

     Evaluate the function  f:

     >>> LogInverse()( np.e )
     -0.6321205588285

      Compute the result element-wise sum for vector inputs :

     >>> LogInverse(w=[1, 2, 3], gamma=3)( [1, 3, np.e] )
     2.01507810453865

     Compute the proximity operator at a given point :

     >>> LogInverse(w=[1,2,3]).prox(  [-2, 3, 4 ])
     array([0.41421356, 3.30277564, 4.23606798])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the
     function 'gamma*f'

     >>> LogInverse( [1,2,3,3], gamma=2.5).prox( [-2, 3, 4, np.e ] )
     array([1.3218996 , 3.95255089, 4.83733009, 3.86652778])
    """

    def __init__(self, w: float or np.ndarray = 1, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.size(w) > 1 and (not isinstance(w, np.ndarray)):
            w = np.array(w)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array) must "
                + "be strictly positive"
            )

        self.gamma = gamma
        if np.any(w < 0):
            raise ValueError(
                "'w (or all its components if it is an array) must be positive"
            )
        self.w = w

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        self._check(x)
        scale = self.gamma
        w = self.w

        def polynom_phi(t):
            return t**3 - x * t**2 - scale * t - w * scale

        def der_phi(t):
            return 3 * t**2 - 2 * x * t - scale

        prox_x = np.abs(x)
        prox_x = newton_(polynom_phi, fp=der_phi, x0=prox_x, low=1e-16, high=np.inf)
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        self._check(x)
        result = np.zeros_like(1.0 * x)
        mask = x > 0
        ww = self.w
        if np.size(self.w) > 1:
            ww = self.w[mask]
        result[mask] = -np.log(x[mask]) + ww * 1 / x[mask]

        result[x <= 0] = np.inf

        return np.sum(self.gamma * result)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
        if (np.size(self.w) > 1) and (np.size(self.w) != np.size(x)):
            raise ValueError("'w' must be either scalar or the same size as 'x'")
