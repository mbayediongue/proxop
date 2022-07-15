"""
Version : 1.0 (06-08-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class Thresholder:
    r"""Computes the proximity operator and the evaluation of gamma*f.

    Where f is the Thresholder (or Support) function defined as:

                    /  a * x   if x < 0
            f(x) = |   0                        if x = 0         with a <= b
                   \  b * x    otherwise

     'gamma*tau' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function, the output is a scalar (sum of the
    element-wise results ) .

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.

     INPUTS
    ========
     x     - scalar or ND array
     a     - scalar or ND array with the same size as 'x'
     b     - scalar or ND array with the same size as 'x'
     gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]

     ========
     Examples
     ========

     Evaluate the 'direct' function f (i.e compute f(x)  ):

     >>> Thresholder(-2, 2)(3)
     6
     >>> Thresholder(-2, 2, gamma=2)([3, 4, -2])
     36

     Compute the proximity operator at a given point :

     >>> Thresholder(-2, 2).prox( 3)
     1
     >>> Thresholder(-1, 2).prox([ -3., 1., 6.])
     array([-2.,  0.,  4.])

     Use a scale factor 'gamma'>0 to commute the proximity operator of gamma*f

     >>> Thresholder(-1, 2, gamma=2).prox([ -3., 1., 6.])
      array([-1.,  0.,  2.])
    """

    def __init__(
        self,
        a: float or np.ndarray = -1,
        b: float or np.ndarray = 1,
        gamma: float or np.ndarray = 1,
    ):
        if np.size(a) > 1 and (not isinstance(a, np.ndarray)):
            a = np.array(a)
        if np.size(b) > 1 and (not isinstance(b, np.ndarray)):
            b = np.array(b)
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it is an array)"
                + " must be strictly positive"
            )
        self.gamma = gamma
        if np.any(a > b):
            raise Exception("'b' must be greater than 'a'")
        self.a = a
        self.b = b

    def prox(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        scale = self.gamma
        self._check_a_b(x)
        return np.minimum(0, x - self.a * scale) + np.maximum(0, x - self.b * scale)

    def __call__(self, x: np.ndarray) -> float:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        self._check_a_b(x)
        return np.sum(
            self.gamma * (self.a * np.minimum(0, x) + self.b * np.maximum(0, x))
        )

    def _check_a_b(self, x):
        if (np.size(self.gamma) > 1) and (np.shape(self.gamma) != np.shape(x)):
            raise Exception("'gamma' must be either scalar or the same size as 'x'")
        if (np.size(self.a) > 1) and (np.shape(self.a) != np.shape(x)):
            raise Exception("'a' must be either scalar or the same size as 'x'")
        if (np.size(self.b) > 1) and (np.shape(self.b) != np.shape(x)):
            raise Exception("'b' must be either scalar or the same size as 'x'")
