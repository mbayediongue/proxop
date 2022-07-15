"""
Version : 1.0 ( 06-14-2022).

DEPENDENCIES:
     - 'pava_python.py' in the folder 'utils'

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.pava_python import pava_python


class MonotoneCone:
    """Compute the projection and the indicator of the monotone cone.

    The 'monotone cone' is the set containing all vectors X that verify:

                               x1 <= x2 <= ... <= xN


     where (1, ..., 1) is a ND array with all components equal to one,
     and (1,..., 1).T denotes its transpose

     INPUTS
    ========
     x    - ND array
    """

    # proximal operator (i.e. projection on the monotone cone)
    def prox(self, x: np.ndarray) -> np.ndarray:
        return pava_python(x)

    # indicator of the monotone cone
    def __call__(self, x: np.ndarray) -> float:
        """
        Indicate if the input 'x' is in the constraint set or not
        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        0      if the input 'x' is in the set
        +inf   otherwise
        """
        if np.size(x) <= 1:
            return 0
        if np.all(x[:-1] <= x[1:]):
            return 0
        return np.inf
