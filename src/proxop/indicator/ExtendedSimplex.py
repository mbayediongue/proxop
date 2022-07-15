"""
Version : 1.0 ( 06-16-2022).

DEPENDENCIES:
     - 'Simplex.py' located in the folder 'indicator'

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.indicator.Simplex import Simplex


class ExtendedSimplex(Simplex):
    r"""Compute the projection onto the extended simplex.

    Where the extended simplex is the set defined as:

       x >= 0 and  (1,..., 1).T * X <= eta


     where (1, ..., 1) is a ND array with all components equal to one,
     and (1,..., 1).T denotes its transpose

     INPUTS
    ========
     x    - ND array
     eta  - positive, scalar or ND array compatible with the blocks of 'x'
    """

    def __init__(
            self,
            eta: float or np.ndarray
    ):
        super().__init__(eta, axis=None)

    # proximal operator (i.e. projection on the extended simplex)
    def prox(self, x: np.ndarray) -> np.ndarray:

        # pre-project onto the positive quarter
        prox_x = np.maximum(x, 0)
        scalar_prod = np.sum(prox_x)
        if scalar_prod <= self.eta:
            return prox_x
        return Simplex(self.eta).prox(x)

    def __call__(self, x: np.ndarray) -> float:
        """
        Indicate if the input 'x' is in the constraint set or not.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        0      if the input 'x' is in the set
        +inf   otherwise
        """

        scalar_prod = np.sum(x)
        if np.all(x >= 0) and (scalar_prod <= self.eta):
            return 0
        return np.inf
