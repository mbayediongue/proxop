"""
Version : 1.0 (27-04-2017), updated (08-06-2022).

Authors  : Giovanni Chierchia and Mbaye Diongue

Copyright (C) 2017

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class Bathtub:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is 'bathtub' (or w-sensitive loss) function defined as:

                f(x) = max{|x|-w,0}

     with w > 0
     'gamma' is the scale factor

    INPUTS
     ========
      x     - scalar or ND array
      w     - positive, scalar or ND array with the same size as 'x'
      gamma - positive, scalar or ND array with the same size as 'x'
              [default value: gamma=1]

     When the input 'x' is an array, the output is computed element-wise :

         -When calling the function (and not the proximity operator) the result
         is computed element-wise SUM. So the command >>>Bathtub()(x) will
         return a scalar even if x is a vector.

         - But for the proximity operator (method 'prox'), the output has the
         same shape as the input 'x'.
         So, the command >>>Bathtub().prox(x)   will return an array with the
         same shape as 'x'

      ========
      Examples
      ========

      Evaluate the 'direct' function :
      >>> Bathtub()(5)
      4
      >>> Bathtub(2, gamma=3)([-4., 2., 3., 5])
      18.0

      Compute the proximity operator :

      >>> Bathtub(2).prox( 3)
      array([2])
      >>> Bathtub(3).prox([ -3., 1., 6.])
      array([-3.,  1.,  5.])

      Use a gamma factor 'gamma'>0 to commute the proximity operator of gamma*f

      >>> Bathtub(1, gamma=2).prox([ -3., 1., 6.])
      array([-1.,  1.,  4.])

      When 'gamma' has the same size as 'x' the computation is done element-wise:
      >>> Bathtub(1, gamma=[1, 2, 3]).prox([ -3., 1., 6.])
      array([-2.,  1.,  3.])
    """

    def __init__(
            self,
            w: float or np.ndarray,
            gamma: float or np.ndarray = 1.0
            ):
        if np.size(w) > 1 and (not isinstance(w, np.ndarray)):
            w = np.array(w)
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array) must"
                + "be strictly positive"
            )

        self.gamma = gamma
        if np.any(w <= 0):
            raise ValueError("'w' must be positive")
        if np.size(w) > 1 and (not isinstance(w, np.ndarray)):
            w = np.array(w)
        if np.size(gamma) > 1 and (not isinstance(gamma, np.ndarray)):
            gamma = np.array(gamma)
        self.w = w
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        self._check(x)
        gamma = self.gamma

        # preliminaries
        abs_x = np.abs(x)
        sign_x = np.sign(x)

        # preliminaries
        abs_x = np.abs(x)
        sign_x = np.sign(x)

        # 2nd branch
        prox_x = self.w * sign_x

        # 1st branch
        mask = abs_x < self.w
        prox_x[mask] = x[mask]

        # 3rd branch
        mask = abs_x > self.w + gamma
        p3 = x - gamma * sign_x
        prox_x[mask] = p3[mask]
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        self.__check(x)
        return np.sum(self.gamma * np.maximum(0, np.abs(x) - self.w))

    def __check(self, x):
        if (not np.isscalar(self.w)) and (
            np.shape(self.w) != np.shape(x) and (len(self.w) > 1)
        ):
            raise ValueError(
                "'w' (or all its components if it is an array) must be either scalar "
                + "or the same size as 'x'"
            )

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise ValueError("gamma' must be either scalar or the same size as 'x'")
        if (np.size(self.w) > 1) and (np.size(self.w) != np.size(x)):
            raise ValueError("'w' must be either scalar or the same size as 'x'")
