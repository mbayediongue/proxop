"""
Version : 1.0 ( 06-10-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""


import numpy as np


class Hyperplane:
    r"""Compute the projection onto the hyperplane.

    Where the hyperplane is the set defined as:

                           u.T * x = mu

     where u.T denotes the transpose of u


     INPUTS
    ========
     x      - ND array
     u      - ND array with the same size as 'x'
     mu     - scalar
    """

    def __init__(
            self,
            u: np.ndarray,
            mu: float
    ):
        self.u = u
        self.mu = mu

    def prox(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the proximal operator (i.e. projection of 'x' on the hyperplane)

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        TYPE : np.ndarray with the same size as x

        """
        self._check(x)
        coef = (self.mu - np.sum(x * self.u)) / (np.linalg.norm(self.u) ** 2)
        return x + coef * self.u

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
        self._check(x)
        tol = 1e-10
        if np.abs(np.sum(x * self.u) - self.mu) <= tol:
            return 0
        return np.inf

    def _check(self, x):
        if (np.shape(self.u)) != np.shape(x):
            raise ValueError("'u' must have the same size as 'x'")
