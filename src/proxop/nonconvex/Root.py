"""
Version : 1.0 (06-18-2022).

DEPENDENCIES:
     'newton.py' in the folder 'utils'

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class Root:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

      where the function f is defined as:


                  f(x) = |x|^q  with  0 < q < 1


    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    tau   - positive, scalar or ND array with the same size as 'x' [default: tau=1]
    q     - in ]0, 1[ , scalar or ND array with the same size as 'x'

    Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>Root(q=0.25, gamma=1)(x) will
    return a scalar even if x is a vector:

    >>> Root(q=0.25, gamma=1)(np.array([-1, 2, 3]))
    3.50528112795521

    But as expected, >>>Root(q=0.25, gamma=1).prox(x)
    will return a vector with the same size as x:

    >>> Root(q=0.25, gamma=1).prox(np.array([-1, 2, 3]))
    array([0.        , 1.8418772 , 2.88712686])
    """

    def __init__(
            self,
            q: float or np.ndarray,
            gamma: float or np.ndarray = 1.0
    ):
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it"
                + " is an array) must be strictly positive"
            )
        if np.any(q <= 0) or np.any(q >= 1):
            raise ValueError(
                "'q' (or all of its components if it"
                + " is an array) must belong to ]0, 1["
            )
        self.gamma = gamma
        self.q = q

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        gamma = self.gamma
        q = self.q

        if np.size(x) <= 1:
            x = np.array([x])
        eps = 1.0e-16
        # select the branches
        alpha = gamma * (np.abs(x) + eps) ** (q - 2)
        mask = alpha <= (2 * (1 - q) / (2 - q)) ** (1 - q) / (2 - q)
        alpha = alpha[mask]
        if np.size(q) > 1:
            q = q[mask]

        # 1st branch
        prox_x = np.zeros(np.shape(x))

        # 2nd branch
        # prepare the Newton's method
        def f(t):
            return t - 1 + alpha * q * t ** (q - 1)

        def df(t):
            return 1 + alpha * q * (q - 1) * t ** (q - 2)

        # use the Newton's method
        root = np.abs(x[mask]) / 2
        low = 5 * 1e-10
        root = newton_(f, df, root, low=low, high=np.inf)

        # explicit solution in the case q==0.5
        mask2 = (q == 0.5) * (alpha <= (2 * (1 - q) / (2 - q)) ** (1 - q) / (2 - q))
        s = (
            2
            * np.sin((np.arccos(3 * np.sqrt(3) * alpha[mask2] / 4) + np.pi / 2) / 3)
            / np.sqrt(3)
        )
        root[mask2] = s**2

        # compute the prox
        prox_x[mask] = root * x[mask]
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.gamma * np.abs(x) ** self.q)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("'gamma' must be either scalar or the same size as 'x'")
