"""
Version : 1.0 ( 06-13-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class Max:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:

                        f(x) = max( x1, ..., xn)

    'gamma' is the scale factor

     INPUTS
    ========
     x         - scalar or ND array
     gamma     - positive, scalar or ND array compatible with the blocks of 'x'
                 [default: gamma=1]
     axis      - None or int, axis of block-wise processing [default: axis=None]
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
                "'gamma' (or all of its components if it is an array)" +
                " must be strictly positive"
            )

        if (axis is not None) and (axis < 0):
            axis = None
        self.axis = axis
        self.gamma = gamma

    def prox(self, x: np.ndarray) -> np.ndarray:
        scale = self.gamma
        self._check(x)
        axis = self.axis
        sz = np.shape(x)

        if np.size(x) <= 1:
            x = np.reshape(x, (-1))

        sz0 = np.shape(x)
        axis = self.axis
        sz0 = np.array(sz0, dtype=int)
        sz0[axis] = 1
        if np.size(scale) > 1:
            scale = np.reshape(scale, sz0)

        # 0. Linearize
        if axis is None:
            x = np.reshape(x, (-1))

        # 1. Order the column elements (sort in decreasing order)
        sort_x = -np.sort(-x, axis=axis)

        # 2. Compute the partial sums: c(j) = ( sum(s(1:j)) - gamma ) / j
        ones_ = np.ones(np.shape(sort_x))

        cum_sum = (np.cumsum(sort_x, axis=axis) - scale) / np.cumsum(ones_, axis=axis)

        # 3. Find the index: n = max{ j \in {1,...,B} : s(j) > c(j) }
        mask = sort_x > cum_sum
        mat = (np.arange(np.size(mask))).reshape(np.shape(mask))
        mask = mask * mat
        ind_max = np.argmax(mask, axis=axis)

        if np.size(ind_max) <= 1:
            prox_x = np.reshape(cum_sum[ind_max], np.shape(scale))
        else:
            prox_x = np.minimum(np.diag(np.take(cum_sum, ind_max,
                                                axis=axis)).reshape(sz0), x)
            prox_x = np.reshape(prox_x, np.shape(x))

        # 4. Compute the prox
        prox_x = np.minimum(x, prox_x)

        # 5. Output is zero if 's(j) > c(j)' for all j
        prox_x = prox_x * ((1 - np.all(mask, axis)).reshape(sz0))

        # 6. revert back
        prox_x = np.reshape(prox_x, sz)

        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        self._check(x)
        return np.sum(self.gamma * np.max(x, axis=self.axis))

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
