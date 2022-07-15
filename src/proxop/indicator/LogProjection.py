"""
Version : 1.0 ( 06-15-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class LogProjection:
    """Compute the projection and the indicator of the set S.

    Where every vector X belonging to S verifies:

         sum_{n=1}^N  alpha * x_n - z_n + z_n * log(z_n/(alpha*x_n)) <= eta


     INPUTS
    ========
     x         - positive ND-array
     eta       - scalar
     z         - positive ND-array with the same size as 'x'
     alpha     - positive, scalar [OPTIONAL, default: alpha=1]
    """

    def __init__(
            self,
            eta: float,
            z: np.ndarray,
            alpha: float = 1
    ):
        if np.size(eta) > 1:
            raise ValueError("'eta' must be a scalar or an array with one element")
        if np.size(alpha) > 1:
            raise ValueError("'alpha' must be a scalar or an array with one element")
        if np.any(z <= 0):
            raise ValueError("'z' (or all its elements) must be strictly positive")
        self.eta = eta
        self.z = z
        self.alpha = alpha

    def prox(self, x: np.ndarray) -> np.ndarray:
        # target function
        alpha = self.alpha
        z = self.z
        eta = self.eta
        if np.any(x <= 0):
            raise ValueError("'x' (or all its elements) must be strictly positive")
        kld = alpha * x - z + z * np.log(z / (alpha * x))

        # projection
        if np.sum(kld) <= eta:  # x is already in the constraint set: projection(x)=x
            return x

        # convert the upper bound
        lamb = self._get_kld_lambda(x, z, alpha, eta)

        # compute the prox
        a = x - alpha * lamb
        prox_x = 0.5 * (a + np.sqrt(a**2 + 4 * lamb * z))
        return prox_x

    def _get_kld_lambda(self, x, z, alpha, eta):
        # prox of KLD (and its derivative)
        def prox_KLD(b):
            return 0.5 * ((x - alpha * b) + np.sqrt((x - alpha * b) ** 2 + 4 * b * z))

        def derive_proxKDL(b):
            return (z - prox_KLD(b)) / np.sqrt((x - alpha * b) ** 2 + 4 * b * z)

        # discrepancy function (and its derivative)
        c = np.sum(z - z * np.log(z))

        def f(b):
            return np.sum(prox_KLD(b) - z * np.log(prox_KLD(b))) - c - eta

        def df(b):
            return np.sum(derive_proxKDL(b) * (1 - z / prox_KLD(b)))

        # find the zero of f
        max_it = 1000
        tol = 1e-7
        lambda_ = 1
        stop = False
        n = 0
        while n < max_it and (not stop):
            lambda_old = lambda_

            # Newton's method
            lambda_ = lambda_ - f(lambda_) / (
                df(lambda_) + tol
            )  # tol is put to avoid dividing by zero

            if np.abs(lambda_ - lambda_old) < tol * np.abs(lambda_old):
                stop = True
            n += 1
        return lambda_

    # Indicator of the constraint set
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
        if np.any(x <= 0):
            raise ValueError("'x' (or all its elements) must be strictly positive")
        kld = self.alpha * x - self.z + self.z * np.log(self.z / (self.alpha * x))
        if np.sum(kld) <= self.eta:
            return 0
        return np.inf
