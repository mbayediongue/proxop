"""
Version : 1.0 ( 06-23-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class BregExp:
    r"""Compute the proximity operator prox_{gamma*f}^{phi}.

    where the functions f and phi are defined as follows:

              f(x)=  exp(x)

              phi(u)= exp(u)

    Note: The function phi is a Legendre type function and induces a Bregman distance

    'gamma * tau' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    - When calling the functions f or phi (method 'phi') the result is computed
    element-wise SUM.
    So the command >>>BregExp()(x) or >>>BregExp().phi(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.So, the command >>>BregExp.prox(x)   will return
    an array with the same shape as 'x'.

    INPUTS
    ========
    x     - scalar or ND array
    gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1]

    =======
    Examples
    ========

    Evaluate the function f:

    >>> BregExp()( np.log(2) )
    2.0
    >>> BregExp()( [-2, 3, 4, np.log(3)] )
    77.81902223956851

     Evaluate The function  phi

     >>> BregExp().phi( [np.log(3)] )
     3.0000000000000004

     Compute the proximity operator at a given point :

     >>> BregExp().prox(  [-2, 3, 4, np.log(3)] )
      array([-2.69314718,  2.30685282,  3.30685282,  0.40546511])
     >>> BregExp(gamma=2).prox(  [-2, 3, 4, np.log(3)] )
     array([-3.09861229,  1.90138771,  2.90138771,  0.        ])
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError("'gamma'  must be strictly positive")
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        return x - np.log(1 + self.gamma)

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.gamma * np.exp(x))

    def phi(self, u: np.ndarray) -> float:
        return np.sum(np.exp(u))

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
