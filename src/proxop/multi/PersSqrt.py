"""
Version : 1.0 ( 06-22-2022).

DEPENDENCIES:
     - 'newton.py' located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class PersSqrt:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:


                /  -sqrt(xi^2 - ||y||_2^2)       if xi > 0 and ||y||_2^2 <= xi
      f(y,xi) = |  0                             if y = 0 and xi = 0
                \  +inf                          otherwise


    'gamma' is the scale factor

    INPUTS
    ========
     y         - scalar or ND array
     xi        - scalar or ND array compatible with the blocks of 'y'
     gamma     - positive, scalar or ND array compatible with the blocks of 'x'
                 [default: gamma=1]
     axis      - None or int, axis of block-wise processing [default: axis=None]
                  axis = None --> 'x' is processed as a single vector [DEFAULT] In this
                  case 'gamma' must be a scalar.
                  axis >=0   --> 'x' is processed block-wise along the specified axis
                  (0 -> rows, 1-> columns ect). In this case 'gamma' and 'xi' must be
                  singletons along 'axis'.
    """

    def __init__(
            self,
            xi: float or np.ndarray = 1,
            gamma: float or np.ndarray = 1,
            axis: int or None = None
            ):
        if np.any(gamma <= 0):
            raise Exception("'gamma'  must be strictly positive")

        if axis is None and np.size(gamma) > 1:
            raise Exception(
                "An 'axis' must be specified when 'gamma' is not a scalar"
            )
        if (axis is not None) and (axis < 0):
            axis = None

        if np.size(gamma) <= 1:
            gamma = np.array([gamma])
        if np.size(xi) <= 1:
            xi = np.reshape(xi, (- 1))

        self.axis = axis
        self.gamma = gamma
        self.xi = xi

    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        scale = self.gamma

        axis = self.axis
        xi = self.xi
        sz0 = np.shape(y)
        if np.size(y) <= 1:
            y = np.array([y])

        sz = np.shape(y)
        sz = np.array(sz, dtype=int)
        sz[axis] = 1

        if np.size(scale) > 1:
            scale = np.reshape(scale, sz)
        if np.size(xi) > 1:
            xi = np.reshape(xi, sz)

        # linearize
        if axis is None:
            y = np.reshape(y, (-1))

        # init
        prox_p = np.zeros(sz0)
        prox_q = np.zeros(sz)

        # condition
        yy = np.linalg.norm(y, ord=2, axis=axis).reshape(sz)
        mask = xi + np.sqrt(scale**2 + yy) > 0
        if np.size(scale) > 1:
            scale = scale[mask]
        if np.size(xi) > 1:
            xi = xi[mask]

        yy_mask = yy[mask]

        # compute t
        def fun_phi(t):
            return 2 * scale * t + xi * t / np.sqrt(1 + t**2) - yy_mask

        def der_fun_phi(t):
            return 2 * scale + (
                xi * np.sqrt(1 + t**2) - xi * t**2 / np.sqrt(1 + t**2)
            ) / (1 + t**2)

        # Finding a root with the Newton method
        t = newton_(fun_phi, fp=der_fun_phi, x0=yy_mask, low=0, high=np.inf)

        coef = np.ones_like(yy)
        coef[mask] = scale * t / yy_mask
        prox_p = y - y * coef
        qq = xi - scale * np.sqrt(1 + t**2)
        prox_q[mask] = qq
        # revert back
        prox_p = np.reshape(prox_p, sz0)
        return [prox_p, prox_q]

    def __call__(self, y: np.ndarray) -> float:
        if np.size(y) <= 1:
            y = np.array([y])
        xi = self.xi
        # evaluate the function
        yy = np.sum(y**2, self.axis)
        if np.any(xi < 0) or np.any(yy > xi**2):
            return np.inf

        p = -np.sqrt(xi**2 - yy)
        return np.sum(self.gamma * p)

    def _multiply_tuple(self, tup):
        res = 1
        for i in tup:
            res = res * i
        return res

    def _check(self, x):
        sz = np.shape(x)
        if len(sz) <= 1:
            self.axis = None
        if len(sz) <= 1 and (np.size(self.gamma) > 1 or np.size(self.xi) > 1):
            raise Exception(
                "'gamma' and 'xi' must be scalar when 'x' is one dimensional"
            )

        if len(sz) > 1 and (self.axis is not None):
            sz = np.array(sz, dtype=int)
            sz[self.axis] = 1
            if np.size(self.gamma) > 1 and (
                self._multiply_tuple(sz) != np.size(self.gamma)
            ):
                raise Exception(
                    "The dimension of 'gamma' is not compatible with the blocks of 'x'"
                )
            if np.size(self.xi) > 1 and (self._multiply_tuple(sz) != np.size(self.xi)):
                raise Exception(
                    "The dimension of 'xi is not compatible with the blocks of 'x'"
                )
