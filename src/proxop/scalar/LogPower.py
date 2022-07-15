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


class LogPower:
    r"""Compute the proximity operator and the evaluation of gamma*f.

     Where f is the 'logarithm+power' function defined as:

                       / ( - log(x) + w * x^ q )     if  x> 0
                f(x)= |
                      \  +INF                                 otherwise

         with w > 0 and q >= 1

     'gamma*tau' is the scale factor

     When the input 'x' is an array, the output is computed element-wise :

     -When calling the function, the output is a scalar (sum of the element-wise
    results ) .

     - But for the proximity operator (method 'prox'), the output has the same shape
     as the input 'x'.

      INPUTS
     ========
      x     - ND array
      q     - positive >=1, scalar or ND array with the same size as 'x'
      w     - positive , scalar or ND array with the same size as 'x'
      gamma - positive, scalar or ND array with the same size as 'x' (default: gamma=1)

     =======
     Examples
     ========

      Evaluate the function  f:

      >>> LogPower(q=1)( 2. )
      1.718281828459045

       Compute the result element-wise for vector inputs :

      >>> LogPower(q=2, w=[1, 2, 3], gamma=3)( [1, 3, np.e] )
      117.2056680243715

      Compute the proximity operator at a given point :

      >>> LogPower(2, [1,2,3]).prox(  [-2, 3, 4 ])
      array([0.33333333, 0.83851648, 0.75951783])

      Use a scale factor 'gamma'>0 to compute the proximity operator of
      the function 'gamma*f'

      >>> LogPower( 2, [1,2,3,3], gamma=2.5).prox( [-2, 3, 4, np.e ] )
      array([0.5       , 0.63221419, 0.5395781 , 0.48925544])

       This is equivalent to use the parameter 'tau' of the method 'prox' :

      >>> LogPower(2, [1,2,3, 3]).prox( [-2, 3, 4, np.e ] , tau=2.5 )
      array([0.5       , 0.63221419, 0.5395781 , 0.48925544])
    """

    def __init__(
        self,
        q: float or np.ndarray,
        w: float or np.ndarray = 1,
        gamma: float or np.ndarray = 1,
    ):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.size(w) > 1 and (not isinstance(w, np.ndarray)):
            w = np.array(w)
        if np.size(q) > 1 and (not isinstance(q, np.ndarray)):
            q = np.array(q)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array) must"
                + "be strictly positive"
            )
        self.gamma = gamma
        if np.any(w < 0):
            raise ValueError(
                "'w (or all its components if it is an array) must"
                + " be strictly positive"
            )
        self.w = w

        if np.any(q < 1):
            raise ValueError(
                "'q' (or all its components if it is an array) must be greater than 1"
            )
        self.q = q

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)

        self._check(x)
        scale = self.gamma
        q = self.q
        w = self.w

        def polynom_phi(t):
            return q * w * scale * t**q + t**2 - x * t - scale

        def derive_polynom_phi(t):
            return q * q * w * scale * t ** (q - 1) + 2 * t - x

        abs_x = np.abs(x)
        prox_x = abs_x / 2

        prox_x = newton_(
            polynom_phi, fp=derive_polynom_phi, x0=prox_x, low=1e-16, high=np.inf
        )

        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        if (np.size(self.w) > 1) and (np.shape(self.w) != np.shape(x)):
            raise ValueError("' w' must be either scalar or the same size as 'x'")

        result = np.zeros_like(1.0 * x)
        mask = x > 0
        ww = self.w
        qq = self.q
        if np.size(self.w) > 1:
            ww = self.w[mask]
        if np.size(self.q) > 1:
            qq = self.q[mask]
        result[mask] = -np.log(x[mask]) + ww * x[mask] ** qq

        result[x <= 0] = np.inf

        return np.sum(self.gamma * result)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
