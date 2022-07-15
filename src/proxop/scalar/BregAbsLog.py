"""
Version : 1.0 ( 06-23-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class BregAbsLog:
    r"""Compute the proximity operator prox_{gamma*f}^{phi}.

    Where the functions f and phi are defined as follows:

              f(x)=  | x - delta |

                      /  - log(u)     if u>0
             phi(u)= |
                     \   +inf         otherwise

    Note: The function phi is a Legendre type function and induces a Bregman distance

    'gamma * tau' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    - When calling the functions f or phi (method 'phi') the result is computed
    element-wise SUM.
    So the command >>>BregAbsLog()(x) or >>>BregAbsLog().phi(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.
    So, the command >>>BregAbsLog.prox(x)   will return an array with the
    same shape as 'x'

    INPUTS
    ========
    x       - scalar or ND array
    delta   - positive, scalar or ND array with the same size as 'x'[default: delta=1]
    gamma   - positive, scalar or ND array with the same size as 'x'[default: gamma=1]

    ========
     Examples
     ========

     Evaluate the  function f :

     >>> BregAbsLog(delta=0.001)(4)
     3.999
     >>> BregAbsLog(delta=0.001)( [-2, 3, 4] )
     8.999

     Evaluate The function  phi

     >> BregAbsLog(delta=0.001).phi(4)
     -1.3862943611198906

     Compute the proximity operator at a given point :

     >>> BregAbsLog(delta=0.1).prox( [-2, 3, 4] )
     array([2.  , 0.75, 0.8 ])
     >>> BregAbsLog(delta=0.1, gamma=2).prox( [-2, 3, 4] )
     array([0.66666667, 0.42857143, 0.44444444])
    """

    def __init__(
            self,
            delta: float or np.ndarray,
            gamma: float or np.ndarray = 1
            ):

        if np.size(delta) > 1 and (not isinstance(delta, np.ndarray)):
            delta = np.array(delta)
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError("'gamma'  must be strictly positive")
        if np.any(delta <= 0):
            raise ValueError("'delta'  must be strictly positive")
        self.gamma = gamma
        self.delta = delta

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        scale = self.gamma
        delta = self.delta
        self._check(x)

        # scalar-like inputs handling
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        # 1st branch
        prox_x = x / (1 + scale * x)

        # 2nd branch
        mask1 = x <= delta / (1 + scale * delta)
        eps = 1.51e-15
        mask = np.logical_and(mask1, (x >= delta / (1 - scale * delta + eps)))
        dd = delta
        if np.size(dd) > 1:
            dd = dd[mask]
        prox_x[mask] = dd

        # 3rd branch
        mask = np.logical_not(mask1)
        gg = scale
        if np.size(gg) > 1:
            gg = gg[mask]
        prox_x[mask] = x[mask] / (1 + gg * x[mask])

        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        return np.sum(self.gamma * np.abs(x - self.delta))

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
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
        if (np.size(self.delta) > 1) and (np.size(self.delta) != np.size(x)):
            raise ValueError("delta' must be either scalar or the same size as 'x'")
