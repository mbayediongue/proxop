"""
Version : 1.0 ( 06-23-2022).

DEPENDENCIES:
     -'lambert_W.py' in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.lambert_W import lambert_W


class BregLpNorm:
    r"""Compute the proximity operator prox_{scale*f}^{phi}.

    Where the functions f and phi are defined as follows:

               f(x)= (1/p)*|x|**p


                         /  u*log(u)-u     if u>0
              phi(u) =  |
                        \   +inf           otherwise

    Note: The function phi is a Legendre type function and induces a Bregman distance

    'gamma * tau' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    - When calling the functions f or phi (method 'phi') the result is computed
    element-wise SUM. So the command >>>BregLpNorm()(x) or >>>BregLpNorm().phi(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the
    same shape as the input 'x'. So, the command >>>BregLpNorm.prox(x)
    will return an array with the same shape as 'x'


    INPUTS
    ========
    x      - scalar or ND array
    p      - scalar >=1 [default: p=1]
    gamma  - positive, scalar or ND array with the same size as 'x' [default: gamma=1]

    =======
    Examples
    ========

    Evaluate the function f:

     >>> BregLpNorm(p=2)( [1, 2, 3] )
     7.0
     >>> BregLpNorm()( [-2, 3, 4]) # default p=1
     9.0

     Evaluate The function  phi

     >> BregLpNorm().phi( [2, 3, 4] )
     1.227308671603782

     Compute the proximity operator at a given point :

     >>> BregLpNorm().prox(  [-2, 3, 4, np.e] )
     array([ 0.04978707,  7.3890561 , 20.08553692,  5.57494152])
     >>> BregLpNorm(2).prox(  [-2, 3, 4, np.e] )
     array([0.12002824, 2.20794003, 2.92627106, 2.01677976])
    """

    def __init__(self, p: float = 1, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError("'gamma'  must be strictly positive")
        if np.any(p < 1) or np.size(p) > 1:
            raise ValueError("'p'  must be a scalar grater or equal to 1")
        self.gamma = gamma
        self.p = p

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        scale = self.gamma
        self._check(x)
        p = self.p

        if p == 1:
            return np.exp(x - scale)

        lambert_x = np.zeros(np.shape(x))

        u = np.log(scale * (p - 1)) + (p - 1) * x
        # approximation of the lambert_W function when the input is "too big"
        mask = u > 100
        lambert_x[mask] = u[mask] - u[mask] / (1 + u[mask]) * np.log(u[mask])
        # use directly the Lambert_W function otherwise
        mask = np.logical_not(mask)
        lambert_x[mask] = lambert_W(np.exp(u[mask]))

        return (lambert_x / (scale * (p - 1))) ** (1 / (p - 1))

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.gamma * (1 / self.p) * (np.abs(x)) ** self.p)

    def phi(self, u: np.ndarray) -> float:
        if np.size(u) > 1 and (not isinstance(u, np.ndarray)):
            u = np.array(u)
        if np.size(u) <= 1:
            u = np.reshape(u, (-1))
        res = np.zeros(np.shape(u))
        mask = u > 0
        res[mask] = u[mask] * np.log(u[mask]) - u[mask]
        res[u <= 0] = np.inf
        return np.sum(res)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
