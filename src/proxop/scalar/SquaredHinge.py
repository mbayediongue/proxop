"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class SquaredHinge:
    r"""Computes the proximity operator and the evaluation of gamma*f.

    Where f is the 'squared hinge loss' function defined as:

            f(x) = gamma * ( max{ x , 0 } ) ^ 2

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function, the output is a scalar (sum of the
    element-wise results ) .

    - But for the proximity operator (method 'prox'), the output has
    the same shape as the input 'x'.

     INPUTS
    ========
     x     - scalar or ND array
     gamma - positive, scalar or ND array with the same size as 'x' (default: gamma=1)

    =======
    Examples
    ========

     Evaluate the function  f:

     >>> SquaredHinge(2)( 2 )
     4.0

      Compute the result element-wise for vector inputs :

     >>> SquaredHinge( gamma=[1, 2, 3])( [-1, 3, -5] )
     9.0

     Compute the proximity operator at a given point :

     >>> SquaredHinge().prox(  [-2, 3, 4 ] )
     array([-2. ,  1.5,  2. ])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the
     function 'gamma*f'

     >>> SquaredHinge(  gamma=[1,2,3,3]).prox( [-2, 3, 4, 6 ] )
     array([-2. ,  1. ,  1. ,  1.5])
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array) "
                + "must be strictly positive"
            )
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        self._check(x)
        scale = self.gamma
        return np.minimum(x, x / (scale + 1))

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.gamma * 0.5 * (np.maximum(x, 0)) ** 2)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.shape(self.gamma) != np.shape(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
