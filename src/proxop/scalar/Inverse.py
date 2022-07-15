"""
Version : 1.0 (06-09-2022).

DEPENDENCIES:
     'newton.py' in the folder 'utils'

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class Inverse:
    r"""Compute the proximity operator and evaluate the inverse function f.

                  / gamma* 1/x^q if x>0
           f(x)= |
                 \  +INF      otherwise with q>=1

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>Inverse()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.So, the command >>>Inverse().prox(x)   will return an
    array with the same shape as 'x'

     INPUTS
    ========
     x     - scalar or ND array
     q     - positive >= 1, scalar or ND array with the same size as 'x' [default: q=1]
     gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]

    =======
    Examples
    ========

     Evaluate the 'direct' function :

     >>> Inverse()( 4 )
     0.25

      Compute the result element-wise for vector inputs :

     >>> Inverse(q=2, gamma=3)( [4., 2., 0.25] )
     48.9375

     Compute the proximity operator at a given point :

     >>> Inverse(2).prox(  [-2, 3, 4 ])
     array([8.85033504e-01, 3.60403299e+08, 3.60403300e+08])

     Use a scale factor 'gamma'>0 to compute the proximity operator of
     the function 'gamma*f'

     >>> Inverse(3, gamma=2.5).prox( [-2, 3, 4, np.e ] )
     array([1.23403930e+00, 4.66240534e+11, 2.38715153e+11, 4.66240534e+11])
    """

    def __init__(self, q=1, gamma=1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.size(q) > 1 and (not isinstance(q, np.ndarray)):
            q = np.array(q)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array) must "
                + "be strictly positive"
            )
        self.gamma = gamma
        if np.any(q < 1):
            raise ValueError("'q' must be greater than 1")
        self.q = q

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)

        self._check(x)
        q = self.q
        abs_x = np.abs(x)
        scale = self.gamma

        def fun_phi(t):
            return t ** (q + 2) - x * t ** (q + 1) - scale * q

        def der_phi(t):
            return (q + 2) * t ** (q + 1) - (q + 1) * x * t * q

        # starting point
        p_init = abs_x / 2

        # Finding the root of the polynom with the Newton method
        prox_x = newton_(fun_phi, fp=der_phi, x0=p_init, low=0, high=np.inf)

        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        self._check(x)
        if np.size(x) <= 1:
            if x > 0:
                return np.sum(self.gamma * (1 / x**self.q))
            return np.inf

        fun_x = np.zeros_like(1.0 * x)
        mask = x > 0
        gamma_ = self.gamma
        if np.size(self.gamma) > 1:
            gamma_ = self.gamma[mask]
        q = self.q
        q_ = q
        if np.size(q) > 1:
            q_ = q[mask]
        fun_x[mask] = gamma_ * (1 / x[mask] ** q_)
        fun_x[np.logical_not(mask)] = np.inf
        return np.sum(fun_x)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
