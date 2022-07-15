"""
Version : 1.0 (06-18-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class L0L2Norm:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

    Where the function f is defined as:

                                   / 0      if ||x||_2=0
               f(x)= ||x||_2^0 =  |
                                  \  1      otherwise


    'gamma' is the scale factor

    When the input 'x' is an array, the computation can vary as follows:
    - axis = None --> 'x' is processed as a single vector [DEFAULT]
       In this case 'gamma' must be a scalar
    - axis >=0   --> 'x' is processed block-wise along the specified axis

    INPUTS
    ========
    x     - ND array
    gamma - positive, scalar or ND array compatible with the blocks of 'x'
            [default: gamma=1]
    axis   - None or int, axis of block-wise processing [default: axis=None]
            axis = None --> 'x' is processed as a single vector [DEFAULT] In this
            case 'gamma' must be a scalar.
            axis >=0   --> 'x' is processed block-wise along the specified axis
            (0 -> rows, 1-> columns ect. In this case, 'gamma' must be singleton
            along 'axis'.
    """

    def __init__(
            self,
            gamma: float or np.ndarray = 1.0,
            axis: int or None = None
    ):
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its elements) must be strictly positive"
            )
        if axis is None and np.size(gamma) > 1:
            raise ValueError(
                "An 'axis' must be specified when" + " 'gamma' is not a scalar"
            )
        if (axis is not None) and (axis < 0):
            axis = None
        self.axis = axis
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        gamma = self.gamma
        axis = self.axis
        sz0 = np.shape(x)
        if axis is None:
            x = np.reshape(x, (-1))

        sz = np.shape(x)
        axis = self.axis
        sz = np.array(sz, dtype=int)
        sz[axis] = 1
        if np.size(gamma) > 1:
            gamma = np.reshape(gamma, sz)

        l2_x2 = np.sum(x**2, axis=axis).reshape(sz)
        mask = l2_x2 >= 2 * gamma
        prox_x = x * mask
        return np.reshape(prox_x, sz0)

    def __call__(self, x: np.ndarray) -> float:
        axis = self.axis
        if np.size(x) <= 1:
            x = np.array([x])
        if axis is None:
            x = np.reshape(x, (-1))
        p = np.linalg.norm(x, ord=2, axis=axis)
        return np.sum(self.gamma * np.array(p != 0))

    def _check(self, x):
        if np.size(self.gamma) <= 1:
            return
        sz = np.shape(x)
        if len(sz) <= 1:
            self.axis = None
        if len(sz) <= 1:
            raise ValueError("'gamma' must be scalar when 'x' is one dimensional")

        if len(sz) > 1 and self.axis is not None:
            sz = np.array(sz, dtype=int)
            sz[self.axis] = 1
            if np.prod(sz) != np.size(self.gamma):
                raise ValueError(
                    "The dimension of 'gamma' is not "
                    + "compatible with the blocks of 'x'"
                )
