"""
Version : 1.0 ( 06-10-2022).

DEPENDENCIES:
     - 'newton.py' located in the folder 'utils'

Author  : Mbaye DIONGUE

Copyright (C) 2019

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class EpiDistance:
    r"""Compute the projection and the indicator of the epigraph of phi.

    Where phi is the function defined as:

            phi(y) = w * ||y - P_C(y)||_2^q       with q=1 or q=2


     The caller must ensure that 'project_C' uses the same type of processing.

      INPUTS
     ========
      y         - ND array
      ksi        - ND array compatible with the blocks of 'y'
      w         - positive, scalar or ND array with the same size as 'ksi'
      axis       - int or None, axis of block-wise processing [OPTIONAL]
               When the input 'y' is an array, the computation can vary as follows:
              - axis = None --> 'y' is processed as a single vector [DEFAULT]
                            (in this case, 'xi' must be scalar)
              - axis >= 0 --> 'y' is processed block-wise along the specified axis
                            (in this case, 'xi' must be singleton along 'axis')
              - axis < 0 --> 'y' is processed element-by-element.
                            (in this case, 'xi' must be the same size as 'y')
      q         - integer, scalar [OPTIONAL, default: 2]
      project_C - function handle with an argument at least [OPTIONAL, default: 0]
      varargin  - additional parameters for the function 'prox_phi' [OPTIONAL]

    """

    def __init__(
            self,
            ksi: np.ndarray,
            axis: int or None = None,
            project_C=lambda y: 0,
            q: int = 2,
            w: float or np.ndarray = 1.0
    ):

        if np.any(w <= 0):
            raise ValueError("'w' must be positive")
        if np.size(w) > 1 and (np.size(w) != np.size(ksi)):
            raise ValueError(" 'w' must be a scalar or must" +
                             " have the same size as 'ksi'")

        if np.size(ksi) <= 1:
            ksi = np.reshape(ksi, (-1))

        self.w = w
        self.ksi = ksi
        self.axis = axis
        self.q = q
        self.project_C = project_C

    # proximal operator (i.e. the projection on the constraint set)
    def prox(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(y)
        ksi = self.ksi
        w = self.w
        axis = self.axis
        project_C = self.project_C
        q = self.q
        eps = 1e-16
        sz = np.shape(y)

        if np.size(y) <= 1:
            y = np.array([y])
        if self.axis is None:
            y = np.reshape(y, (-1))

        # preliminaries
        py = project_C(y)
        dy = np.linalg.norm(y - py, ord=2, axis=axis)  # distance separating y from C

        if q == 1:
            t = np.maximum(0, dy + w * ksi) / (1 + w**2)
        else:
            # Finding the root of with the Newton's method
            def fun_phi(t):
                return (
                    q * w**2 * t ** (2 * q - 1)
                    - q * w * ksi * t ** (q - 1)
                    + t
                    - np.abs(y)
                )

            def derive_phi(t):
                return (
                    (2 * q - 1) * q * w**2 * t ** (2 * q - 2)
                    - (q - 1) * q * w * ksi * t ** (q - 2)
                    + 1
                )

            # starting point
            low = (np.maximum(ksi, 0) / w) ** (1 / q)
            root_init = 3 * low

            t = newton_(fun_phi, fp=derive_phi, x0=root_init, low=low, high=np.inf)

        # 3rd branch
        prox_p = py + t / (dy + eps) * (y - py)

        prox_t = w * np.linalg.norm(prox_p - project_C(prox_p), ord=2, axis=axis) ** q

        # default cas: axis==None => y is processed as single vector
        if axis is None:
            if dy == 0 and ksi < 0:
                return [y, np.zeros_like(ksi)]
            elif w * dy**q <= ksi:
                return [y, ksi]
            else:
                return [prox_p, prox_t]

        # 2nd branch
        mask_p = np.ones(np.shape(y), dtype=bool)
        mask_t = np.logical_and(dy == 0, ksi < 0)
        mask_p = mask_t * mask_p
        prox_p[mask_p] = y[mask_p]
        prox_t[mask_t] = 0

        # 1st branch

        mask_t = w * dy**q <= ksi
        mask_t = np.logical_and(dy == 0, ksi < 0)
        mask_p = mask_t * mask_p
        prox_p[mask_p] = y[mask_p]
        prox_t[mask_t] = ksi[mask_t]

        prox_p = np.reshape(prox_p, sz)
        return [prox_p, prox_t]

    # indicator of the constraint set
    def __call__(self, y: np.ndarray) -> float:
        self._check(y)
        TOL = 1e-10
        if self.axis is None:
            y = np.reshape(y, (-1))
        py = self.project_C(y)
        dy = np.linalg.norm(y - py, ord=2, axis=self.axis)

        if np.all(self.w * dy**self.q <= self.ksi + TOL):
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
