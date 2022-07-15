"""
Version : 1.0 ( 06-09-2022).

DEPENDENCIES :
     lambert_W.py - located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.lambert_W import lambert_W


class Entropy:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the entropy function defined as:


                      /  x*log(x)            if  x> 0
               f(x)= |   0                              if x =0
                     \  +INF                           otherwise


     When the input 'x' is an array, the output is computed element-wise.

     'gamma' is the scale factor

     When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>Entropy(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'. So, the command >>>Entropy.prox(x)   will return
    an array with the same shape as 'x'.

      INPUTS
     ========
     x     - scalar or ND array-like
     gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]

    =======
    Examples
    ========

     Evaluate the 'direct' function :

     >>> Entropy()( np.e )
     2.718281828459045

     Element-wise sum when the input is a vector:
     >>> Entropy()( [1, 0, 3, np.e] )
      6.01411869446337

     Compute the proximity operator at a given point :

     >>> Entropy().prox(  [-2, 3, 4, np.e] )
     array([0.04747849, 1.5571456 , 2.20794003, 1.38940572])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the
     function 'gamma*f' :

     >>> Entropy(gamma=2).prox( [-2, 3, 4, np.e ] )
     array([0.12700814, 1.        , 1.37015388, 0.90903194])
    """

    def __init__(
            self,
            gamma: float or np.ndarray = 1
            ):

        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its elements if it is an array)"
                + "must be strictly positive"
            )
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        self._check(x)
        scale = self.gamma
        inv_scale = 1 / scale
        return scale * lambert_W(inv_scale * np.exp(inv_scale * x - 1))

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        result = np.zeros(np.shape(x))
        mask = x > 0
        result[mask] = x[mask] * np.log(x[mask])
        mask2 = x < 0
        result[mask2] = np.inf
        return np.sum(self.gamma * result)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
