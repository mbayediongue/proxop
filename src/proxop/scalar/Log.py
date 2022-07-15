"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class Log:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is defined as:

                      /  - log(x)          if  x> 0
               f(x)= |
                     \    +INF              otherwise


    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>Log()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'. So, the command >>>Log().prox(x)   will
    return an array with the same shape as 'x'

     INPUTS
    ========
     x     - scalar or ND array
     gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]

    =======
    Examples
    ========

     Evaluate the function  f:

     >>> Log()( np.e )
     -1.0

      Compute the result element-wise for vector inputs :

     >>> Log(gamma=3)( [1, 3, np.e] )
     -6.2958368660043

     Compute the proximity operator at a given point :

     >>> Log().prox(  [-2, 3, 4 ])
     array([0.41421356, 3.30277564, 4.23606798])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the function
     'gamma*f'

     >>> Log( gamma=2.5).prox( [-2, 3, 4, np.e ] )
     array([0.87082869, 3.67944947, 4.54950976, 3.44415027])
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array)"
                + " must be strictly positive"
            )
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        self._check(x)
        return 0.5 * (x + np.sqrt(x**2 + 4 * self.gamma))

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        self._check(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        result = np.zeros_like(x, dtype=np.float)
        gamma = self.gamma
        mask = x > 0
        g_ = gamma
        if np.size(gamma) > 1:
            g_ = gamma[mask]
        result[mask] = -g_ * np.log(x[mask])
        result[x <= 0] = np.inf
        return np.sum(result)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
