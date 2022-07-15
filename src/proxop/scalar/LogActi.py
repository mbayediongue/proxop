"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class LogActi:
    r"""Compute the proximity operator and the evaluation of f.

    Where f is the logarithmic activation function defined as:

            f(x) = cosh(x)-(x^2)/2

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>LogActi()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.
    So, the command >>>LogActi().prox(x)   will return an array with the same
    shape as 'x'

     INPUTS
    ========
     x     - scalar or ND array

     =======
     Examples
     ========

     >>> LogActi()( -3 )
      11.585536923187668
     >>> LogActi()( [1, -1.5, -0.5] )
      1.0986921694972378

     Compute the proximity operator at a given point :

     >>> LogActi().prox(-3)
     -1.3862943611198906
     >>> LogActi().prox(  [-2, 3, -0.5] )
     array([-1.09861229,  1.38629436, -0.40546511])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log(1 + np.abs(x))

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        return np.sum(np.exp(np.abs(x)) - np.abs(x) - 1 - 0.5 * x**2)
