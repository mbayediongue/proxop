"""
Version : 1.0 (06-09-2022).

DEPENDENCIES:
     'Thresholder.py' in the folder scalar

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.scalar.Thresholder import Thresholder


class HingeLoss(Thresholder):
    r"""Computes the proximity operator and the evaluation of gamma*f.

    Where f is the hinge Loss function defined as:

                      f(x) = max{x,0}

    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>HingeLoss()(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the
    same shape as the input 'x'. So, the command >>>HingeLoss().prox(x)
    will return an array with the same shape as 'x'

     INPUTS
    ========
     x     - ND array
     gamma  - positive, scalar or array with the same size as 'x' [default: gamma=1]

    =======
    Examples
    ========

     Evaluate the 'direct' function :

     >>> HingeLoss()
     0.5

    The output is computed as element-wise sum for vector inputs :

     >>> HingeLoss(gamma=2)
     6.5

     Compute the proximity operator at a given point :

     >>> HingeLoss().prox(  [-2, 3, 4 ])
     array([-2,  2,  3])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the function
     'gamma*f':

     >>> HingeLoss(gamma=2.5).prox( [-2, 3, 4, np.e ] )
     array([-2.        ,  0.5       ,  1.5       ,  0.21828183])
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        super().__init__(0, 1, gamma)
