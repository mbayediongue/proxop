"""
Version : 1.0 (06-10-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class PRelu:
    r"""Computes the proximity operator and the evaluation of f.

    Where f is the 'PRelu activation' function defined as:

                          /   0                            if  x> 0
                   f(x)= |
                         \   (1/alpha +1) * x^2 /2         otherwise


    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function, the output is a scalar (sum of the element-wise
    results ).

    - But for the proximity operator (method 'prox'), the output has the
    same shape as the input 'x'.


    Note: The proximity operator of the function defined below is the PRelu function

     INPUTS
    ========
     x       - scalar or ND array
     alpha   - positive, scalar or ND array with the same size as 'x'  in ]0, 1]
                [default value: alpha=0.25]

    =======
    Examples
    ========

     Evaluate the function  f:

     >>> PRelu()( -2 )
      10.0

      Compute the result element-wise sum for vector inputs :

     >>> PRelu(alpha=0.5)( [-1, -3, 2] )
     15.0

     Compute the proximity operator at a given point :

     >>> PRelu().prox(  [-2, 3, 4 ])
      array([-0.5,  3. ,  4. ])
     >>> PRelu( [0.5, 0.25, 0.3]).prox( [-2, 3, 4 ] )
     array([-1.,  3.,  4.])
    """

    def __init__(self, alpha: float or np.ndarray = 0.25):
        if np.size(alpha) > 1 and (not isinstance(alpha, np.ndarray)):
            alpha = np.array(alpha)
        if np.any(alpha <= 0) or np.any(alpha > 1):
            raise ValueError(
                "'alpha (or all its components if it is an array) must be in ]0, 1]"
            )
        self.alpha = alpha

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        if np.isscalar(x):
            if x > 0:
                return x
            else:
                return self.alpha * x
        prox_x = self.alpha * x
        mask = x > 0
        prox_x[mask] = prox_x[mask]
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        fun_x = np.zeros(np.shape(x))
        mask = x <= 0
        alpha_ = self.alpha
        if np.size(self.alpha) > 1:
            alpha_ = self.alpha[mask]

        fun_x[mask] = (1 / alpha_ + 1) * x[mask] ** 2 / 2

        return np.sum(fun_x)
