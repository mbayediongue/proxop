"""
Version : 1.0 ( 06-16-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class EpiMax:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:

                phi(y)= w * max( x1, ..., x_n)

     INPUTS
    ========
     y    - ND array
     ksi  - ND array compatible with the blocks of 'y' (see above)
     w    - positive, scalar or ND array with the same size as 'ksi' [default: w=1]
     axis - int or None, direction of block-wise processing
            When the input 'y' is an array, the computation can vary as follows:
            - axis = None --> 'y' is processed as a single vector [DEFAULT]
                                   (in this case, 'xi' must be scalar)
            - axis >= 0 --> 'y' is processed block-wise along the specified axis
            (in this case, 'xi' must be singleton along 'axis')
    """

    def __init__(
            self,
            ksi: np.ndarray,
            axis: int or None = None,
            w: float or np.ndarray = 1.0
    ):

        if np.any(w <= 0):
            raise ValueError("'w' must be positive")
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or have the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.array([ksi])

        self.w = w
        self.axis = axis
        self.ksi = ksi

    # proximal operator (i.e. the projection on the constraint set)
    def prox(self, y):
        self._check(y)
        ksi = self.ksi
        w = self.w
        axis = self.axis
        if len(np.shape(y)) == 1 and np.size(ksi) <= 1:
            axis = None

        if axis is not None:
            if axis == 0:
                ksi = np.reshape(ksi, (2 * axis + 1) * (1, -1))
            else:
                ksi = np.reshape(ksi, (-1, 1))
        # sort inputs
        v = y * w
        v = np.sort(v, axis=axis)

        # compute the cumulative sum
        cum_sum = np.concatenate((w**2 * ksi, np.flip(v, axis=axis)), axis=axis)
        cum_sum = np.cumsum(cum_sum, axis=axis)
        cum_sum = np.flip(cum_sum, axis=axis)

        axis_ = axis
        if axis is None:
            axis_ = 0
        # Normalize...
        cum_sum = cum_sum / (
            w**2 + np.shape(v)[axis_] + 1 - np.cumsum(np.ones_like(cum_sum), axis=axis)
        )

        infty = np.inf * np.ones(np.shape(ksi))
        v_low = np.concatenate((-infty, v), axis=axis)
        v_high = np.concatenate((v, infty), axis=axis)

        mask = np.logical_and(v_low < cum_sum, cum_sum <= v_high)

        sz_ = np.shape(mask)[axis_]
        mat = (np.arange(np.size(mask)) % sz_).reshape(np.shape(mask))
        mask = mask * mat

        ind_max = np.max(mask, axis=axis)

        if axis is not None:
            prox_t = np.reshape(cum_sum[ind_max], np.shape(ksi))
        else:
            prox_t = (np.reshape(np.diag(np.take(cum_sum, ind_max, axis=axis)),
                                 np.shape(ksi)))

        prox_y = np.minimum(y, prox_t / w)

        return [prox_y, prox_t]

    # indicator of the constraint set
    def __call__(self, y) -> float:
        self._check(y)
        TOL = 1e-10
        if self.axis is None:
            y = np.reshape(y, (-1))
        mask = self.w * np.max(y, axis=self.axis) <= self.ksi + TOL

        if np.all(mask):
            return 0
        return np.inf

    def _check(self, y):
        sz_ = 1
        if self.axis is not None:
            sz = np.array(np.shape(y), dtype=int)
            sz_ = np.delete(sz, self.axis)
        cond1 = self.axis is not None and np.any(sz_ != np.shape(self.ksi))
        cond2 = self.axis is None and np.any(np.size(self.ksi) != sz_)
        if cond1 or cond2:
            try:
                self.ksi = np.reshape(self.ksi, sz_)
            except ValueError:
                print("'ksi' is not compatible with the size of 'y' ")
