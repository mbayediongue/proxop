"""
Version : 1.0 ( 06-09-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class ArgsinhActi:
    r"""Compute the proximity operator and the evaluation of f.

    where f is the inverse hyperbolic sine activation function:

                    f(x) = cosh(x) -(x^2)/2

    When the input 'x' is an array, the output is computed element-wise :

    - When calling the function  the result is computed element-wise SUM.
    So, the command >>>Thresholder()(x) will return a scalar even if x is a
    vector.

    - But for the proximity operator (method 'prox'), the output has the
    same shape as the input 'x'.So, the command >>>Thresholder().prox(x)
    will return an array with the same shape as 'x'


     INPUTS
    ========
     x     - scalar or ND array

    ========
    Examples
    ========

    Evaluate the 'direct' function :
    >>> ArgsinhActi()(-0.5)
     1.0026259652063807
    >>>ArgsinhActi()([-2., -0.5, 0, 0.25, 3])
    10.332646751947351

    Compute the proximity operator at a given point :
    >>> ArgsinhActi().prox(3)
     1.8184464592320668s
    >>> ArgsinhActi().prox([ -3., 0.5, 6.])
    array([-1.81844646,  0.48121183,  2.49177985])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        return np.arcsinh(x)

    def __call__(self, x: np.ndarray) -> float:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return np.sum(np.cosh(x) - 0.5 * x**2)
