"""
Version : 1.0 (06-07-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class Huber:
    r"""Compute the proximity operator and evaluate of the gamma*f.

    Where f is the Huber Loss function definded as :


                  /  1/2 * x^2             if |x| <= w
           f(x)= |
                 \  w* |x| - w^2  /2       otherwise


                 with w>0


    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>Huber()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'. So, the command >>>Huber.prox(x)   will return an array
    with the same shape as 'x'.

    ========
     x      - ND array
     w      - positive, scalar or ND array with the same size as 'x'
     gamma  - positive, scalar or ND array with the same size as 'x'[default: gamma=1]

    =======
    Examples
    ========

     Evaluate the 'direct' function :

     >>> Huber(2)( 0.25 )
      0.03125

      Compute the result element-wise for vector inputs :

     >>> Huber(2)( [-1, 3, 0.25] )
     4.53125

     Compute the proximity operator at a given point :

     >>> Huber(3).prox(  [-2, 3, 4 ])
      array([-1,  1,  2])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the
     function 'gamma*f'

     >>> Huber(3, gamma=2.5).prox( [-2, 3, 4, np.e ] )
     array([-0.57142857,  0.85714286,  1.14285714,  0.77665195])
    """

    def __init__(
            self,
            w: float or np.ndarray,
            gamma: float or np.ndarray = 1
            ):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it is an array) must"
                + " be strictly positive"
            )
        self.gamma = gamma
        if np.size(w) > 1 and (not isinstance(w, np.ndarray)):
            w = np.array(w)
        if np.any(w <= 0):
            raise Exception(
                "'w' (or all of its components if it is an array) must"
                + " be strictly positive"
            )
        self.w = w

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        self._check(x)
        gamma = self.gamma
        w = self.w

        # preliminaries
        abs_x = np.abs(x)
        sign_x = np.sign(x)

        # 2nd branch
        prox_x = x - gamma * w * sign_x

        # 1st branch
        mask = abs_x <= w * (gamma + 1)
        gamma_ = gamma
        if np.size(gamma) > 1:
            gamma_ = gamma[mask]
        prox_x[mask] = x[mask] / (1 + gamma_)
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        abs_x = np.abs(x)
        w = self.w
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            if abs_x <= w:
                return np.sum(self.gamma * (0.5 * x**2))
            return np.sum(self.gamma * (w * abs_x - w**2 / 2))

        fun_x = 0.5 * x**2
        mask = abs_x > w
        if np.size(w) > 1:
            fun_x[mask] = abs_x[mask] * w[mask] - w[mask] ** 2 / 2
        else:
            fun_x[mask] = abs_x[mask] * w - w**2 / 2
        return np.sum(self.gamma * fun_x)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise Exception("gamma' must be either scalar or the same size as 'x'")
        if (np.size(self.w) > 1) and (np.size(self.w) != np.size(x)):
            raise Exception("w' must be either scalar or the same size as 'x'")
