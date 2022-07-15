"""
Version : 1.0 (06-09-2022).

DEPENDENCIES:
    - 'Thresholder.py' located in the folder 'scalar'

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.scalar.Thresholder import Thresholder


class AbsValue(Thresholder):
    """Compute the proximity operator and the evaluation of the gamma*f.

      where the function f is defined as:

                        f(x) = |x|

     'gamma' is the scale factor

     When the input 'x' is an array, the output is computed element-wise :

     -When calling the function (and not the proximity operator) the result
     is computed element-wise SUM. So the command >>>AbsValue()(x) will
     return a scalar even if x is a vector.

     -But for the proximity operator (method 'prox'), the output has the same
     shape as the input 'x'.
     So, the command >>>AbsValue().prox(x)   will return an array with the same
     shape as 'x'


      INPUTS
     ========
     x     - scalar or ND array
     gamma - positive, scalar or ND array with the same size as 'x'
     [default: gamma=1]


     ========
     Examples
     ========

     >>> AbsValue()(-3)
     3
     >>> AbsValue(gamma=2)(3)
     6

     >>> AbsValue().prox( 3)
     2
     >>> AbsValue().prox([ -3., 1., 6.])
     array([-2.,  0.,  5.])


    Use a scale factor 'gamma'>0 to compute the proximity operator of  the
    function 'gamma*f'

     >>> AbsValue(gamma=2).prox([ -3., 1., 6.])
      array([-1.,  0.,  4.])
    """

    def __init__(self, gamma: float or np.ndarray = 1.0):
        super().__init__(-1, 1, gamma)
