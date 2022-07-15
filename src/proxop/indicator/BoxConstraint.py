"""
Version : 1.0 ( 06-10-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class BoxConstraint:
    r"""Compute the projection onto the box constraint set.

                           low <= x <= high

     INPUTS
    ========
     x    - ND array
     low  - scalar or ND array with the same size as 'x'
     high - scalar or ND array with the same size as 'x'
    """

    def __init__(
            self,
            low: float or np.ndarray,
            high: float or np.ndarray
    ):
        if np.any(low > high):
            raise Exception("'low' must be lower than 'high'")
        self.low = low
        self.high = high

    # proximal operator (i.e. projection on the box)
    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        return np.maximum(self.low, np.minimum(x, self.high))

    # indicator of the box
    def __call__(self, x: np.ndarray) -> float:
        """Indicate if the input 'x' is in the box or not.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        0      if the input 'x' is in the box
        +inf   otherwise
        """
        self._check(x)
        if np.all(self.low <= x) and np.all(x <= self.high):
            return 0
        return np.inf

    def _check(self, x: np.ndarray):
        if (np.size(self.low) > 1) and (np.shape(self.low) != np.shape(x)):
            raise Exception("'low' must be either scalar or the same size as 'x'")

        if (np.size(self.high) > 1) and (np.shape(self.high) != np.shape(x)):
            raise Exception("'high' must be either scalar or the same size as 'x'")
