"""
Version : 1.0 ( 06-23-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class BregBoltzShannon:
    r"""Compute the proximity operator prox_{gamma*f}^{phi}.

    Where the functions f and phi are defined as follows:

                        /  x*log(x) - w*x      if x>0
               f(x) =  |
                       \  +inf                 otherwise


                         /  u*log(u)-u     if u>0
              phi(u) =  |
                        \   +inf           otherwise

    Note: The function phi is a Legendre type function and induces a Bregman distance

    'gamma' is the scale factor

     When the input 'x' is an array, the output is computed element-wise :

    - When calling the functions f or phi (method 'phi') the result is computed
    element-wise SUM. So the command >>>BregBoltzShannon()(x) or
    >>BregBoltzShannon().phi(x) will give a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'. So, the command >>>BregBoltzShannon.prox(x)
    will return an array with the same shape as 'x'


    INPUTS
    ========
    x      - scalar or ND array
    w      - scalar or ND array with the same size as 'x'  [default: w=1]
    gamma  - positive, scalar or ND array with the same size as 'x' [default: gamma=1]

     ========
     Examples
     ========

     Evaluate the  function f :

     >>> BregBoltzShannon()(1)
     -1.0
     >>> BregBoltzShannon(w=0)( [0.2, 3, 4, np.e] )
     11.237408556456117

     Evaluate The function  phi

     >> BregBoltzShannon(w=0).phi( [0.2, 3, 4, np.e] )
     1.3191267279970713

     Compute the proximity operator at a given point :

     >>> BregBoltzShannon(2).prox( [-2, 3, 4] )
     array([  0.36787944,  54.59815003, 148.4131591 ])
     >>> BregBoltzShannon(w=1, gamma=2).prox( [-2, 3, 4] )
     array([ 0.13533528, 20.08553692, 54.59815003])
    """

    def __init__(
            self,
            w: float or np.ndarray = 1,
            gamma: float or np.ndarray = 1
            ):
        if np.size(w) > 1 and (not isinstance(w, np.ndarray)):
            w = np.array(w)
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError("'gamma'  must be strictly positive")
        self.gamma = gamma
        self.w = w

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        scale = self.gamma
        self._check(x)
        w = self.w
        return np.exp((x + scale * (w - 1)))

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        self._check(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        res = np.zeros(np.shape(x))
        mask = x > 0
        w = self.w
        if np.size(w) > 1:
            w = w[mask]
        res[mask] = x[mask] * np.log(x[mask]) - w * x[mask]
        res[x <= 0] = np.inf
        return np.sum(self.gamma * res)

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
        if (np.size(self.w) > 1) and (np.size(self.w) != np.size(x)):
            raise ValueError("'w' must be either scalar or the same size as 'x'")
