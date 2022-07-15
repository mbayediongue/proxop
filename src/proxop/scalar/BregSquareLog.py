"""
Version : 1.0 ( 06-23-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class BregSquareLog:
    r"""Compute the proximity operator prox_{gamma*f}^{phi}.

    Where f and phi are defined as follows:

               f(x)= (1/2)*x^2

                         /  - log(u)      if u>0
              phi(u) =  |
                        \   +inf           otherwise

    Note: The function phi is a Legendre type function and induces a Bregman distance

    'gamma * tau' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    - When calling the functions f or phi (method 'phi') the result is computed
    element-wise SUM. So the command >>>BregSquareLog()(x) or >>>BregSquareLog.phi(x)
    will return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.
    So, the command >>>BregSquareLog.prox(x)   will return an array with the same
    shape as 'x'

    INPUTS
    ========
    x      - scalar or ND array
    gamma  - positive, scalar or ND array with the same size as 'x' [default: gamma=1]


    =======
    Examples
    ========

    Evaluate the function f:

     >>> BregSquareLog()( -3 )
     4.5
     >>> BregSquareLog()( [1, 2, 3] )
     7.0

     Evaluate The function  phi

     >> BregSquareLog().phi( [2, 3, 4] )
     -3.1780538303479453

     Compute the proximity operator at a given point :

     >>> BregSquareLog().prox(  [-2, 3, 4, np.e] )
     array([-0.78077641,  0.84712709,  0.88278222,  0.83283647])
     >>> BregSquareLog().prox(  [-2, 3, 4, np.log(2)] )
     array([-0.78077641,  0.84712709,  0.88278222,  0.51167407])
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise Exception("'gamma'  must be strictly positive")
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        scale = self.gamma
        self._check(x)
        x[x == 0] = 1e-20
        return (np.sqrt(1 + 4 * scale * x**2) - 1) / (2 * scale * x)

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        return np.sum(self.gamma * 0.5 * x**2)

    def phi(self, u: np.ndarray) -> float:
        if np.size(u) > 1 and (not isinstance(u, np.ndarray)):
            u = np.array(u)
        if np.size(u) <= 1:
            u = np.reshape(u, (-1))
        res = np.zeros(np.shape(u))
        mask = u > 0
        res[mask] = -np.log(u[mask])
        res[u <= 0] = np.inf
        return np.sum(res)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise Exception("gamma' must be either scalar or the same size as 'x'")
