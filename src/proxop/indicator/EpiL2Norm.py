"""
Version : 1.0 ( 06-16-2022).

Authors  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class EpiL2Norm:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:

                phi(y)= w * ||y||_2

     

     INPUTS
    ========
     y    - ND array
     ksi  - ND array compatible with the blocks of 'y' (see above)
     w    - positive, scalar or ND array with the same size as ksi [default: w=1.0]
     axis - int or None, direction of block-wise processing
            When the input 'y' is an array, the computation can vary as follows:
            - axis = None --> 'y' is processed as a single vector [DEFAULT]
                                   (in this case, 'xi' must be scalar)
            - axis >= 0 --> 'y' is processed block-wise along the specified axis
            (in this case, 'xi' must be singleton along 'axis')
    """

    def __init__(
            self,
            ksi: float or np.ndarray,
            axis: int or None = None,
            w: float or np.ndarray = 1.0
    ):
        if np.any(w <= 0):
            raise ValueError("'w'  must be positive")
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or have the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.reshape(ksi, (-1))

        self.w = w
        self.axis = axis
        self.ksi = ksi

    # proximal operator (i.e. the projection on the constraint set)
    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        w = self.w
        axis = self.axis
        eps = 1e-16
        sz = np.shape(y)
        if self.axis is None:
            y = np.reshape(y, (-1))

        yy = np.linalg.norm(y, ord=2, axis=axis)
        # branch 3
        a = np.maximum(0, 1 + w * ksi / (yy + eps)) / (1 + w**2)
        # branch 1
        a[w * yy <= ksi] = 1
        # branch 2
        a[np.logical_and(yy == 0, ksi < 0)] = 0

        # compute the projection
        prox_y = a * y
        prox_t = np.maximum(ksi, w * a * yy)

        # reshape
        prox_y = np.reshape(prox_y, sz)
        return [prox_y, prox_t]

    # indicator of the constraint set
    def __call__(self, y: np.ndarray) -> float:
        self._check(y)
        TOL = 1e-10
        if self.axis is None:
            y = np.reshape(y, (-1))
        mask = self.w * np.linalg.norm(y, ord=2, axis=self.axis) <= self.ksi + TOL
        if np.all(mask):
            return 0
        return np.inf

    def _check(self, y: np.ndarray or float):
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
