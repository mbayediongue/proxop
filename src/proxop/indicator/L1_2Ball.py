"""
Version : 1.0 ( 06-14-2022).

DEPENDENCIES:
     - 'L1Ball.py' - located in the folder 'indicator'
     - 'Max.py' in the folder 'multi'

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""


import numpy as np
from proxop.indicator.L1Ball import L1Ball


class L1_2Ball:
    """Compute the projection and the indicator of the constraint set S.

    Where every vector X belonging to S verifies:

         ||x^(1)||_2 + ... + ||x^(B)||_2 <= eta


     INPUTS
    ========
     x         - ND array
     eta       - positive scalar
     axis - int or None, direction of block-wise processing
            When the input 'x' is an array, the computation can vary as follows:
            - axis = None --> 'x' is processed as a single vector [DEFAULT]
            - axis >= 0 --> 'x' is processed block-wise along the specified axis
               (axis=0 -> rows, axis=1 -> columns etc.).
    """

    def __init__(
            self,
            eta: float,
            axis: int or None = None
    ):
        if np.size(eta) > 1 or eta <= 0:
            raise ValueError("'eta' must be a positive scalar")
        self.eta = eta
        self.gamma = 1.0
        self.axis = axis

    def prox(self, x):
        xa = np.linalg.norm(x, ord=2, axis=self.axis, keepdims=True)
        TOL = 1e-10
        # compute the projection
        if (
            np.sum(xa) <= self.eta + TOL
        ):  # the vector if already in the set => it is equal to its projection
            return x

        # project the block norms
        beta = L1Ball(self.eta)(xa)

        # compute the scale factors
        ya = beta / xa
        ya[xa < 1e-16] = 0

        # rescale the blocks
        prox_x = ya * x
        return prox_x

    def __call__(self, x):
        """Indicate if the input 'x' is in the constraint set or not.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        0      if the input 'x' is in the set
        +inf   otherwise
        """
        l2_norm_x = np.linalg.norm(x, ord=2, axis=self.axis, keepdims=True)
        if np.sum(l2_norm_x) <= self.eta:
            return 0
        return np.inf
