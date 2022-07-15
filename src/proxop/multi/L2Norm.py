"""
Version : 1.0 ( 06-13-2022).

DEPENDENCIES:
    -'newton.py'  - located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""""

import numpy as np


class L2Norm:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the euclidian norm:

                        f(x) = ||x||_2

    'gamma' is the scale factor

     INPUTS
    ========
     x         - ND array
     gamma     - positive, scalar or ND array compatible with the blocks of 'x'
                 [default: gamma=1]
     axis    - None or int, axis of block-wise processing [default: axis=None]
                  axis = None --> 'x' is processed as a single vector [DEFAULT] In this
                  case 'gamma' must be a scalar.
                  axis >=0   --> 'x' is processed block-wise along the specified axis
                  (0 -> rows, 1-> columns ect. In this case, 'gamma' must be singleton
                  along 'axis'.
    """

    def __init__(self,
                 gamma: float or np.ndarray = 1,
                 axis: int or None = None
                 ):

        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array)" +
                " must be strictly positive"
            )
        if (axis is not None) and (axis < 0):
            axis = None
        self.gamma = gamma
        self.axis = axis

    def prox(self, x: np.ndarray) -> np.ndarray:

        scale = self.gamma
        self._check(x)

        xx = np.sqrt(np.sum(x**2, axis=self.axis, keepdims=True))
        eps = 1e-16  # to avoid dividing by zeros
        xx = np.maximum(0, 1 - scale / (eps + xx))
        p = x * xx
        return p

    def __call__(self, x: np.ndarray) -> float:
        self._check(x)
        l2 = np.sqrt(np.sum(x**2, axis=self.axis))

        return np.sum(self.gamma * l2)

    def _check(self, x):
        scale = self.gamma
        if np.size(scale) <= 1:
            return
        if self.axis is None:
            raise ValueError(
                "'gamma' must be a scalar when the argument 'axis' is equal to 'None'"
            )

        sz = np.shape(x)
        if len(sz) <= 1:
            self.axis = None

        if len(sz) <= 1:
            raise ValueError(
                "'tau' and 'gamma' must be scalar when 'x' is one dimensional"
            )

        if len(sz) > 1 and (self.axis is not None):
            sz = np.array(sz, dtype=int)
            sz[self.axis] = 1
            if np.size(scale) > 1 and (np.prod(sz) != np.size(scale)):
                raise ValueError(
                    "The dimension of 'tau' or 'gamma' is not compatible" +
                    " with the blocks of 'x'"
                )