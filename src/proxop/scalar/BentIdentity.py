"""
Version : 1.0 ( 06-07-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class BentIdentity:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the Bent identity function defined as:

                 /   x/2 - ln( x+ 1/2)/4     if  x > -1/2
          f(x)= |
                \  +INF                      otherwise

     When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>BentIdentity()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.
    So, the command >>>BentIdentity().prox(x)   will return an array with
    the same shape as 'x'


     INPUTS
    ========
     x     - scalr or ND array

     ========
     Examples
     ========

    Evaluate the 'direct' function :

    >>> BentIdentity()(-0.5)
    array([inf])
    >>> BentIdentity()([-0.2, 15, 0, 0.25, 0.3])
    7.591776396181667

    Compute the proximity operator at a given point :

    >>> BentIdentity().prox(3)
    2.58113883008419
    >>> BentIdentity().prox([ -3., 0.5, 6.])
    array([-0.41886117,  0.30901699,  5.54138127])
    """

    def prox(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        prox_x = (x + np.sqrt(x**2 + 1) - 1) / 2
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if np.any(x <= -0.5):
            return np.inf
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        result = x / 2 - np.log(x + 1 / 2) / 4
        return np.sum(result)
