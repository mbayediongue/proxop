"""
Version : 1.0 (06-18-2022).

DEPENDENCIES:
     'solver_cubic.py' in the folder 'utils'

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""


import numpy as np
from proxop.utils.solver_cubic import solver_cubic


class Cauchy:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

      where the function f is defined as:

               f(x) = log( delta + x^2)


    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise.

     INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]
    delta -positive, scalar or ND array with the same size as 'x'[default: delata=1e-10]

    Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>Cauchy(delta=0.01, gamma=1)(x)
    will return
    a scalar even if x is a vector:

    >>> Cauchy(0.01, 1)(np.array([-1., 2., 3.]))
    3.597076643791892

    But as expected, >>>Cauchy(delta=0.01, gamma=1).prox(x) will
    return a vector with the same size as x:

    >>> Cauchy(0.01, 1).prox(np.array([-1., 2., 3.]))
    array([-0.00498744,  0.01005025,  0.0152717 ])
    """

    def __init__(
        self,
        delta: float or np.ndarray = 1e-10,
        gamma: float or np.ndarray = 1.0
    ):
        if np.any(gamma <= 0):
            raise Exception("'gamma' (or all of its elements)" +
                            " must be strictly positive")
        if np.any(delta <= 0):
            raise Exception(
                "'delta' (or all of its elements) must be strictly positive"
            )
        self.gamma = gamma
        self.delta = delta

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        gamma = self.gamma
        delta = self.delta

        # Solve the cubic equation
        prox_x, p2, p3 = solver_cubic(1, -x, delta + 2 * gamma, -x * delta)

        def f(t):
            return 0.5 * np.abs(x - t) ** 2 + gamma * np.log(delta + t**2)

        mask = np.logical_and(np.isreal(p2), f(np.real(p2)) < f(prox_x))
        prox_x[mask] = np.real(p2[mask])

        mask = np.logical_and(np.isreal(p2), f(np.real(p3)) < f(prox_x))
        prox_x[mask] = np.real(p3[mask])

        return np.reshape(prox_x, np.shape(x))

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.gamma * np.log(self.delta + x**2))

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise Exception("'gamma' must be either scalar or the same size as 'x'")
