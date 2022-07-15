"""
Version : 1.0 ( 06-14-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class PositiveL2Ball:
    """Compute the projection and the indicator of the positive L2-ball.

    Recall: every vector X belonging to the positive L2-ball verifies:

                        ||x||_2 <= eta and x >= 0

     INPUTS
    ========
     x         - ND array
     eta       - positive scalar
     axis      - integer, axis of block-wise processing [default: axis=None]
               (axis=0 -> rows, axis=1 -> columns etc.)
    """

    def __init__(
            self,
            eta: float,
            axis: int or None = None
    ):
        if np.size(eta) > 1 or eta <= 0:
            raise Exception("'eta' must be a positive scalar")
        self.eta = eta
        self.gamma = 1.0
        self.axis = axis

    def prox(self, x: np.ndarray) -> np.ndarray:
        xplus = np.maximum(x, 0)
        l2_x_plus = np.linalg.norm(xplus, ord=2, axis=self.axis)
        if l2_x_plus <= self.eta:
            return xplus
        return self.eta / l2_x_plus * xplus

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
        l2_x = np.linalg.norm(x, ord=2, axis=self.axis, keepdims=True)
        if np.all(l2_x <= self.eta) and np.all(x >= 0):
            return 0
        return np.inf
