"""
Version : 1.0 (06-09-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class FermiDiracEntropy:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the Fermi Dirac Entropy function defined as:

                      /  ( x*log(x)  + (1-x)log(x) )    if  0 < x < 1
               f(x)= |  0                               if x in {0, 1}
                     \  +INF                            otherwise


    'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>FermiDIracEntropy(x) will
    return a scalar even if x is a vector.

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'. So, the command >>>FermiDIracEntropy.prox(x)
    will return an array with the same shape as 'x'.


    INPUTS
    ========
     x     - scalar or array
     gamma  - positive, scalar or array with the same size as 'x' [default: gamma=1]

     =======
    Examples
    ========

     Evaluate the 'direct' function :

     >>> FermiDiracEntropy()( 0.5 )
     -0.6931471805599453

      Compute the result element-wise sum for vector inputs :

     >>> FermiDiracEntropy(gamma=2)( [1, 0, 0.25, 0.5] )
     -2.510964650357507

     Compute the proximity operator at a given point :

     >>> FermiDiracEntropy().prox(  [-2, 3, 4, np.e] )
     array([0.10829336, 0.89170664, 0.9545842 , 0.86456313])

     Use a scale factor 'gamma'>0 to compute the proximity operator of  the function
     'gamma*f' :

     >>> FermiDiracEntropy(gamma=2.5).prox( [-2, 3, 4, np.e ] )
     array([0.28609273, 0.71390727, 0.78356663, 0.69220157])
    """

    def __init__(self, gamma: float or np.ndarray = 1):
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array) must be"
                + " strictly positive"
            )
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        gamma = self.gamma
        self._check(x)

        limit = 8
        # RUN HALEY'S METHOD FOR SOLVING w = W_{exp(x/gamma)}(exp(x/gamma)/gamma)
        # where W_r(x) is the generalized Lambert function

        w = np.zeros_like(x)
        igamma = 1 / gamma
        c = x * igamma - np.log(gamma)
        z = np.exp(c)
        r = np.exp(x * igamma)

        # ASYMPTOTIC APPROX
        approx = 1 - np.exp((1 - x) * igamma)

        # INITIALIZATION
        # Case 1: gamma <= 1/30
        mask = np.logical_and(z > 1, gamma <= 1 / 30)
        w[mask] = c[mask] - np.log(c[mask])
        # Case 2: gamma > 1/30
        mask = np.logical_and(z > 1, gamma > 1 / 30)
        if np.size(igamma) > 1:
            igamma = igamma[mask]
        w[mask] = igamma * approx[mask]

        # RUN
        maxiter = 20
        test_end = np.zeros_like(x)
        tol = 1e-8
        epsilon = 1e-16  # to avoid dividing by zero
        n = np.size(x)
        for _ in range(maxiter):
            e = np.exp(w)
            y = w * e + r * w - z
            v = e * (1 + w) + r
            u = e * (2 + w)
            wnew = w - y / (v - y * u / (2 * v))

            mask = np.abs(wnew - w) / (np.abs(w) + epsilon) < tol
            test_end[np.logical_and(mask, test_end == 0)] = 1
            not_mask = np.logical_not(mask)
            idx_update = np.logical_and(not_mask, test_end == 0)
            w[idx_update] = wnew[idx_update]  # the rest stays constant !
            if np.sum(test_end) == n:  # stop !
                break

        prox_x = gamma * w
        if np.size(prox_x) <= 1:
            if c > limit and gamma > 1:
                prox_x = approx
            return np.minimum(prox_x, 1)

        # ASYMPTOTIC DVP
        mask_approx = np.logical_and(c > limit, gamma > 1)
        prox_x[mask_approx] = approx[mask_approx]

        # FINAL TRESHOLD TO AVOID NUMERICAL ISSUES FOR SMALL GAMMA
        prox_x = np.minimum(prox_x, 1)
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        if np.size(x) > 1 and (not isinstance(x, np.ndarray)):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        result = np.zeros(np.shape(x))
        mask = np.logical_and(x > 0, x < 1)
        result[mask] = x[mask] * np.log(x[mask]) + (1 - x[mask]) * np.log(1 - x[mask])
        mask2 = np.logical_or(x < 0, x > 1)
        result[mask2] = np.inf
        return np.sum(self.gamma * result)

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
