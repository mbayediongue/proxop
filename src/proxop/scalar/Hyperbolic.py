"""
Version : 1.0 (06-07-2022).

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


class Hyperbolic:
    r"""Compute the proximity operator and evaluate of gamma*f.

    Where f is the hyperbolic function defined as:

               f(x) = sqrt ( x^2 + delta^2)

    'gamma' is the scale factor

     When the input 'x' is an array, the output is computed element-wise :

     -When calling the function (and not the proximity operator) the result
     is computed element-wise SUM. So the command >>>Hyperbolic()(x) will
     return a scalar even if x is a vector.

     - But for the proximity operator (method 'prox'), the output has the
     same shape as the input 'x'.So, the command >>>Hyperbolic().prox(x)   will
     return an array with the same shape as 'x'

     INPUTS
     ========
     x      - scalar or array
     delta  - strictly positive, scalar or array with the same size as 'x'
     gamma  - positive, scalar or array with the same size as 'x' [ default: gamma=1]

     =======
     Examples
     ========

     Evaluate the 'direct' function :

     >>> Hyperbolic(delta=3)( 4 )
     5.0

     Compute the result as element-wise sum for vector inputs :

     >>> Hyperbolic(delta=2, gamma=3)( [-1, 3, 0.25] )
     23.571551070115248

     Compute the proximity operator at a given point :

     >>> Hyperbolic(3).prox(  [-2, 3, 4 ])
     array([-1.54269095,  2.37870272,  3.26376922])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the
     function 'gamma*f'

     >>> Hyperbolic(3, gamma=2.5).prox( [-2, 3, 4, np.e ] )
     array([-1.12333244,  1.74370688,  2.42744522,  1.5630981 ])
    """

    def __init__(
            self,
            delta: float or np.ndarray,
            gamma: float or np.ndarray = 1
            ):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all its components if it is an array)"
                + " must be strictly positive"
            )
        if np.any(delta <= 0):
            raise ValueError(
                "'delta' (or all its components if it is an array)"
                + " must be strictly positive"
            )
        self.delta = delta
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        scale = self.gamma
        self._check(x)
        abs_x = np.abs(x)

        def fun_phi(p):
            return (
                p**4
                + (-2 * abs_x) * p**3
                + (x**2 - scale**2 + self.delta**2) * p**2
                + (-2 * abs_x * self.delta**2) * p
                + (self.delta**2) * (x**2)
            )

        def der_phi(p):
            return (
                4 * p**3
                - 6 * abs_x * p**2
                + 2 * (x**2 - scale**2 + self.delta**2) * p
                + (-2 * abs_x * self.delta**2)
            )

        # starting point
        root = abs_x / 2
        root = newton_(fun_phi, fp=der_phi, x0=root, low=0, high=abs_x)

        return root * np.sign(x)

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        return np.sum(self.gamma * np.sqrt(x**2 + self.delta**2))

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
