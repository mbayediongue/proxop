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


class Power:
    r"""Computes the proximity operator and the evaluation of gamma*f.

    Where f is the power function defined as:

              f(x) = x ^ q

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function, the output is a scalar (sum of the element-wise results)

    - But for the proximity operator (method 'prox'), the output has the same shape
    as the input 'x'.

     INPUTS
    ========
     x     - scalar or ND array
     q     -  scalar >= 1
     gamma - positive, scalar or ND array with the same size as 'x' (default: gamma=1)

    =======
    Examples
    ========

     Evaluate the function  f:

     >>> Power(3)( 2 )
     8

      Compute the result element-wise for vector inputs :

     >>> Power(q=2, gamma=[1, 2, 3])( [-1, 3, 2] )
     31

     Compute the proximity operator at a given point :

     >>> Power(2).prox(  [-2, 3, 4 ])
     array([-0.66666667,  1.        ,  1.33333333])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the
     function 'gamma*f'

     >>> Power( 2, gamma=[1,2,3,3]).prox( [-2, 3, 4, 6 ] )
     array([-0.66666667,  0.6       ,  0.57142857,  0.85714286])
    """

    def __init__(self, q: float, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it is an array)"
                + " must be strictly positive"
            )
        self.gamma = gamma
        if np.any(q < 1) or np.size(q) > 1:
            raise Exception("'q' must be a scalar greater than 1")
        self.q = q

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)

        self._check(x)
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        q = self.q
        gamma = self.gamma

        if q == 1:
            prox_x = sign_x * np.maximum(0, abs_x - gamma)

        elif q == 4 / 3:
            ksi = np.sqrt(x**2 + 256 * gamma**3 / 729)
            ksi_minus = np.power(np.sqrt(ksi - x), 1 / 3)
            ksi_plus = np.power(np.sqrt(ksi + x), 1 / 3)

            prox_x = x + 4 * gamma / (3 * 2 ** (1 / 3)) * (ksi_minus - ksi_plus)

        elif q == 3 / 2:
            gamma2 = gamma**2
            prox_x = x + 9 / 8 * gamma * 2 * sign_x * (
                1 - np.sqrt(1 + 16 * abs_x / (9 * gamma2))
            )

        elif q == 2:
            prox_x = x / (2 * gamma + 1)

        elif q == 3:
            prox_x = sign_x * (np.sqrt(1 + 12 * gamma * abs_x) - 1)

        elif q == 4:
            ksi = np.sqrt(x**2 + 1 / (27 * gamma))
            prox_x = np.power((ksi + x) / (8 * gamma), 1 / 3) - np.power(
                (ksi - x) / (8 * gamma), 1 / 3
            )

        else:

            def polynom_phi(t):
                return gamma * q * t ** (q - 1) + t - abs_x

            def der_phi(t):
                return gamma * q * (q - 1) * t ** (q - 2) + 1

            # starting point
            prox_x = abs_x / 2
            prox_x = newton_(polynom_phi, der_phi, prox_x, 0, np.inf)
            prox_x = sign_x * prox_x

        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.gamma * np.abs(x) ** (self.q))

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.shape(self.gamma) != np.shape(x)):
            raise Exception("gamma' must be either scalar or the same size as 'x'")
