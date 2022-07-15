"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class LogisticLoss:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the logistic loss function defined as:

         f(x)= gamma*log( 1+ exp(x) )

    'gamma' is the scale factor

     When the input 'x' is an array, the output is computed element-wise :

     -When calling the function, the output is a scalar (sum of the element-wise
    results ).

     - But for the proximity operator (method 'prox'), the output has the same shape
     as the input 'x'.

      INPUTS
     ========
      x     - ND array
      gamma - positive, scalar or ND array with the same size as 'x' (default: gamma=1)

     =======
     Examples
     ========

      Evaluate the function  f:

      >>> LogisticLoss()( 0 )
       0.6931471805599453

       Compute the result element-wise sum for vector inputs :

      >>> LogisticLoss(gamma=3)( [-1, 3, np.e] )
      18.432098909466088

      Compute the proximity operator at a given point :

      >>> LogisticLoss().prox(  [-2, 3, 4 ])
      array([-2.10829336,  2.10829336,  3.0454158 ]))

      Use a scale factor 'gamma'>0 to compute the proximity operator of
      the function 'gamma*f'

      >>> LogisticLoss( gamma=2.5).prox( [-2, 3, 4, np.e ] )
      array([-1.04044264, -0.68339574, -0.55797119, -0.70389204])
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array)"
                + "must be strictly positive"
            )
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        self._check(x)
        scale = self.gamma

        x = (
            x / scale
        )  # we do this scaling to use the same method as in FermiDiracEntropy.py class
        gamma = 1 / scale

        limit = 8
        # RUN HALEY'S METHOD TO SOLVE w = W_{exp(x/gamma)}(exp(x/gamma)/gamma)
        # where W_r(x) is the generalized Lambert function

        w = np.zeros_like(1.0 * x)
        inv_gamma = 1 / gamma
        c = x * inv_gamma - np.log(gamma)
        z = np.exp(c)
        r = np.exp(x * inv_gamma)

        # ASYMPTOTIC APPROX
        approx = 1 - np.exp((1 - x) * inv_gamma)

        # INITIALIZATION
        # Case 1: gamma <= 1/30
        mask = np.logical_and(z > 1, gamma <= 1 / 30)
        w[mask] = c[mask] - np.log(c[mask])
        # Case 2: gamma > 1/30
        mask = np.logical_and(z > 1, gamma > 1 / 30)
        if np.size(inv_gamma) > 1:
            inv_gamma = inv_gamma[mask]
        w[mask] = inv_gamma * approx[mask]

        # RUN
        max_iter = 20
        test_end = np.zeros(np.shape(x))
        tol = 1e-8
        epsilon = 1e-20  # to avoid dividing by zero
        n = np.size(x)

        for _ in range(max_iter):

            e = np.exp(w)
            y = w * e + r * w - z
            v = e * (1 + w) + r
            u = e * (2 + w)
            w_new = w - y / (v - y * u / (2 * v))

            mask = np.abs(w_new - w) / (epsilon + np.abs(w)) < tol
            test_end[np.logical_and(mask, test_end == 0)] = 1
            not_mask = np.logical_not(mask)
            idx_update = np.logical_and(not_mask, test_end == 0)
            w[idx_update] = w_new[idx_update]  # the rest stays constant !
            if np.sum(test_end) == n:  # stop !
                break

        prox_x = gamma * w

        if np.size(prox_x) <= 1:
            if c > limit and gamma > 1:
                prox_x = approx
            return x - scale * np.minimum(prox_x, 1)

        # ASYMPTOTIC DVP
        mask_approx = np.logical_and(c > limit, gamma > 1)
        prox_x[mask_approx] = approx[mask_approx]

        # FINAL THRESHOLD TO AVOID NUMERICAL ISSUES FOR SMALL GAMMA
        prox_x = np.minimum(prox_x, 1)

        return x - scale * prox_x

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(self.gamma * np.log(1 + np.exp(x)))

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
