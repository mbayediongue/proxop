"""
Version : 1.0 ( 06-16-2022).

DEPENDENCIES:
     - 'EpiMax.py' located in the folder 'indicator'

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.indicator.EpiMax import EpiMax


class EpiLinfNorm:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:

                phi(y)= w * ||y||_inf

    where ||.||_inf denotes the infinity norm

     INPUTS
    ========
     y    - ND array
     ksi  - ND array compatible with the blocks of 'y' (see above)
     w    - positive, scalar or ND array with the same size as 'ksi' [default: w=1.0]
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
            raise ValueError(
                "'w' (or all of its components if it is an array) must be positive"
            )
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or must have" +
                             " the same size as 'ksi'")
        if np.size(ksi) <= 1:
            ksi = np.array([ksi])

        self.w = w
        self.axis = axis
        self.ksi = ksi

    # proximal operator (i.e. the projection on the constraint set)
    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        w = self.w
        axis = self.axis

        abs_y = np.abs(y)
        prox_t = EpiMax(axis=axis, ksi=ksi, w=w).prox(abs_y)[1]
        prox_t = np.maximum(0, prox_t)

        # Compute p
        prox_p = np.sign(y) * np.minimum(abs_y, prox_t / w)

        return [prox_p, prox_t]

    # indicator of the constraint set
    def __call__(self, y: np.ndarray) -> float:
        self._check(y)
        TOL = 1e-10
        if self.axis is None:
            y = np.reshape(y, (-1))
        mask = self.w * np.max(np.abs(y), axis=self.axis) <= self.ksi + TOL
        if np.all(mask):
            return 0
        return np.inf

    def _check(self, y: np.ndarray):
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
