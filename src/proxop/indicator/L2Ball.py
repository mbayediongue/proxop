"""
Version : 1.0 ( 06-14-2022).

DEPENDENCIES:
     - 'L2Norm.py'    - located in the folder 'multi'

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.multi.L2Norm import L2Norm


class L2Ball:
    """Compute the projection and the indicator of the L2 ball.

    Recall: every vector X belonging to the L2 ball verifies:

                        ||x||_2 <= eta

     INPUTS
    ========
     x         - ND array
     eta       - positive scalar
     axis - int or None, direction of block-wise processing
            When the input 'x' is an array, the computation can vary as follows:
            - axis = None --> 'x' is processed as a single vector [DEFAULT]
            - axis >= 0 --> 'x' is processed block-wise along the specified axis
              (axis=0 -> rows, axis=1 -> columns etc.).  In this case, 'eta'
    """

    def __init__(
            self,
            eta: float,
            axis: int or None = None
    ):
        if np.size(eta) > 1 or eta <= 0:
            raise Exception("'eta' must be a positive scalar")
        self.eta = eta
        self.axis = axis

    def prox(self, x: np.ndarray) -> np.ndarray:
        return x - L2Norm(gamma=self.eta, axis=self.axis).prox(x)

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
        TOL = 1e-10
        l2_x = np.linalg.norm(x, ord=2, axis=self.axis, keepdims=True)
        if np.all(l2_x <= self.eta + TOL):
            return 0
        return np.inf
