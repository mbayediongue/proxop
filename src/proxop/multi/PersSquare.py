"""
Version : 1.0 ( 06-21-2022).

DEPENDENCIES:
     - 'solver_cubic.py' located in the folder 'multi'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.solver_cubic import solver_cubic


class PersSquare:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:


                /   ||y||_2^2 / xi          if xi > 0
      f(y,xi) = |  0                        if y = 0 and xi = 0
                \  +inf                     otherwise


            'gamma * tau' is the scale factor

    When the input 'y' is an array, the computation can vary as follows:
    - axis= None --> 'x' is processed as a single vector [DEFAULT]
             ( In this case 'gamma' must be a scalar )
    - axis >=0   --> 'y' is processed block-wise along the specified axis
             (in this case, 'xi' must be singleton along 'axis')
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
                  (0 -> rows, 1-> columns ect. In this case 'gamma' and 'xi' must be
                  singletons along 'axis'.
    """
    
    def __init__(
            self,
            xi: float or np.ndarray = 1,
            gamma: float or np.ndarray = 1,
            axis=None
            ):
        if np.any(gamma <= 0):
            raise Exception("'gamma'  must be strictly positive")

        if axis is None and np.size(gamma) > 1:
            raise Exception(
                "A 'axis' must be specified when 'gamma' is not a scalar"
            )
        if (axis is not None) and (axis < 0):
            axis = None

        if np.size(gamma) <= 1:
            gamma = np.reshape(gamma, (-1))
        if np.size(xi) <= 1:
            xi = np.reshape(xi, (-1))

        self.axis = axis
        self.gamma = gamma
        self.xi = xi
        self._check_done = False

    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        scale = self.gamma

        axis = self.axis
        xi = self.xi
        sz0 = np.shape(y)
        if np.size(y) <= 1 or axis is None:
            y = np.reshape(y, (-1))

        sz = np.shape(y)
        sz = np.array(sz, dtype=int)
        sz[axis] = 1

        if np.size(scale) > 1:
            scale = np.reshape(scale, sz)
        if np.size(xi) > 1:
            xi = np.reshape(xi, sz)

        # 3rd branch
        l2_x = np.sqrt(np.sum(y**2, axis=axis)).reshape(sz)
        t = solver_cubic(scale, 0, 4 * xi + 8 * scale, -8 * l2_x)[0]
        # t= np.max( np.real(t), axis=axis).reshape(sz)
        pp = scale * t / l2_x
        qq = scale * t**2 / 4

        # 1st branch
        mask = l2_x**2 <= -4 * scale * xi

        pp[mask] = 1
        if np.size(xi) > 1:
            qq[mask] = -xi[mask]
        else:
            qq[mask] = -xi

        # 2nd branch
        mask = np.logical_and(l2_x == 0, xi > 0).reshape(sz)
        pp[mask] = 1
        qq[mask] = 0

        # compute the prox
        prox_p = y - y * pp
        prox_q = xi + qq

        # revert back
        prox_p = np.reshape(prox_p, sz0)

        return [prox_p, np.reshape(prox_q, (-1))]

    def __call__(self, y: np.ndarray) -> float:
        if np.size(y) <= 1:
            y = np.reshape(y, (-1))

        if np.any(self.xi < 0):
            return np.inf
        l2_y2 = np.sum(y**2, axis=self.axis)
        fun_y = l2_y2 / self.xi
        fun_y[self.xi == 0] = 0

        return np.sum(self.gamma * fun_y)

    def _check(self, x):
        if self.check_done or (np.size(self.gamma) <= 1 and np.size(self.xi) <= 1):
            return

        sz = np.shape(x)
        if len(sz) <= 1:
            self.axis = None
        if len(sz) <= 1 and (np.size(self.gamma) > 1 or np.size(self.xi) > 1):
            raise Exception(
                "'gamma' and 'xi' must be scalars when 'x' is one dimensional"
            )

        if len(sz) > 1 and self.axis is not None:
            sz = np.array(sz, dtype=int)
            sz[self.axis] = 1

            if np.size(self.gamma) > 1 and (
                np.prod(sz) != np.size(self.gamma)
            ):
                raise Exception(
                    "The dimension of 'gamma' is not compatible with the blocks of 'x'"
                )
            if np.size(self.xi) > 1 and (np.prod(sz) != np.size(self.xi)):
                raise Exception(
                    "The dimension of 'xi is not compatible with the blocks of 'x'"
                )

        self._check_done = True
