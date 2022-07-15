"""
Version : 1.0 (06-08-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class GMActi:
    r"""Compute the proximity operator of the Geman MacClure activation function.

    The Geman MacClure activation function is definded as:

           /  mu*arctan( sqrt( |x|/(mu - |x|) ) -  sqrt( |x|(mu - |x|) x^2   if |x|<mu
    f(x)= |  mu(pi-mu)/2                                                     if |x|=mu
          \  +INF                                                            otherwise

        with mu = 8/(3*sqrt(3))

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>GMActi()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'. So, the command >>>GMActi().prox(x)   will return
    an array with the same shape as 'x'.

     INPUTS
    ========
     x     - scalar or array

     =======
    Examples
    ========

     Evaluate the 'direct' function :

     >>> GMActi()( 0.5 )
      0.073622336567425

      Compute the result element-wise for vector inputs :

     >>> GMActi()( [-1, 0, 0.25] )
     0.39239604176

     Compute the proximity operator at a given point :

     >>> GMActi().prox(  [-2, 3, 4, np.e] )
     array([-1.23168057,  1.38564065,  1.44903597,  1.35607581])
    """

    def __init__(self):
        self.mu = 8 / (3 * np.sqrt(3))

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        mu = self.mu
        return mu * np.sign(x) * x**2 / (1 + x**2)

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        mu = self.mu
        abs_x = np.abs(x)

        result = mu * (np.pi - mu) / 2 * np.ones(np.shape(x))
        result[np.abs(x) > mu] = np.inf
        mask = np.abs(x) < mu
        abs_x = abs_x[mask]
        result[mask] = (
            mu * np.arctan(np.sqrt(abs_x) / (mu - abs_x))
            - np.sqrt(abs_x * (mu - abs_x))
            - x[mask] ** 2 / 2
        )
        return np.sum(result)
