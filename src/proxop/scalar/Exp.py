"""
Version : 1.0 ( 06-08-2022).

DEPENDENCIES :
     lambert_W.py - located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.lambert_W import lambert_W


class Exp:
    r"""Computes the proximity operator and the evaluation of gamma*f.

    Where f is the exponential function:

            f(x)= exp(x)


    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>Exp()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'. So, the command >>>Exp().prox(x)   will
    return an array with the same shape as 'x'

     INPUTS
    ========
    x      -  array_like
    gamma  - positive, scalar or ND array with the same size as 'x'
            [default value: gamma=1]

    =======
    Examples
    ========

     Evaluate the 'direct' function :

     >>> Exp()( np.log(2) )
     2.0

     Compute the result element-wise Sum for vector inputs
     >>> Exp(gamma=2)( [-1, 0, 3, np.log(2)] )
      46.90683272871822

     Compute the proximity operator at a given point :

     >>> Exp().prox(  [-2, 3, 4, np.log(2)] )
     array([-2.12002824,  0.79205997,  1.07372894, -0.15945832])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the function
     'gamma*f':

     >>> Exp(gamma=2.5).prox( [-2, 3, 4, np.log(3) ] )
      array([-2.26069497,  0.13594702,  0.37230635, -0.46761867])
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it is an array) must "
                + "be strictly positive"
            )
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        self._check(x)
        scale = self.gamma
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        prox_x = np.zeros(np.shape(x))
        # we use a taylor approximation to avoid divergence when the input is 'too big'
        u = np.log(scale) + x
        mask = u > 100
        u = u[mask]
        prox_x[mask] = u - u / (1 + u) * np.log(u)
        not_mask = np.logical_not(mask)
        if np.size(scale) > 1:
            scale = scale[not_mask]
        prox_x[not_mask] = lambert_W(scale * np.exp(x[not_mask]))
        return x - prox_x

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.gamma * np.exp(x))

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise Exception("gamma' must be either scalar or the same size as 'x'")
