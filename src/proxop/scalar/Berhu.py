"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class Berhu:
    r"""Compute the proximity operator and the evaluation of the function f.

    Where f is the function defined as:

                  / ( x^2 + w^2 ) / ( 2* W)     if |x| > w
           f(x)= |
                 \  |w|                         otherwise

                 with w>0


    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function the result
    is computed element-wise SUM. So the command >>>Berhu()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same shape
    as the input 'x'.
    So, the command >>>Berhu().prox(x)   will return an array with the same
    shape as 'x'

     INPUTS
    ========
     x     - ND array
     w     - positive, scalar or ND array with the same size as 'x'

     ========
     Examples
     ========

     Evaluate the 'direct' function:

     >>> Bh=Berhu(w=2)
     >>> Bh(3)
      3.25
     >> Bh([-1, 2, 3])
     7.25

     Compute the proximity operator at a given point :

     >>> Bh.prox( 3)
     array([2])
     >>> Bh.prox([ -3., 1., 6.])
     array([-2.,  0.,  4.])
    """

    def __init__(
            self,
            w: float or np.ndarray
    ):
        if np.size(w) > 1 and (not isinstance(w, np.ndarray)):
            w = np.array(w)
        if np.any(w <= 0):
            raise ValueError(
                "'w' (or all of its components if it is an array) must "
                + "be strictly positive"
            )
        self.w = w

    def prox(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        self._check(x, self.w)
        # preliminaries
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        w = self.w

        # 2nd branch
        prox_x = x - sign_x

        # 1st branch
        mask = abs_x > w + 1

        if np.size(w) > 1:
            prox_x[mask] = w[mask] * x[mask] / (1 + w[mask])
        else:
            prox_x[mask] = w * x[mask] / (w + 1)

        # third branch
        mask2 = abs_x <= 1
        prox_x[mask2] = 0

        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        self._check(x, self.w)
        abs_x = np.abs(x)
        w = self.w

        result = np.abs(w) * np.ones(np.shape(x))
        mask = abs_x > w

        if np.size(w) > 1:
            result[mask] = (w[mask] ** 2 + x[mask] ** 2) / (2 * w[mask])

        else:
            result[mask] = (w**2 + x[mask] ** 2) / (2 * w)

        return np.sum(result)

    def _check(self, x, w):
        if (np.size(w) > 1) and (np.size(w) != np.size(x)):
            raise ValueError("'w' must be either scalar or the same size as 'x'")
