"""
Version : 1.0 ( 06-14-2022).

DEPENDENCIES:
    -'newton.py'  - located in the folder 'utils'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
from proxop.utils.newton import newton_


class L2BallBarrier:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:

                   / -log( radius2 -  || x - c||^2    if  || x - c||_2)^2 < radius2
            f(x) =|
                  \   + inf                           otherwise


     INPUTS
    ========
     x       - ND array
     c       - ND array with the same size as 'x' ( center of the ball constraint)
     radius2 - scalar , positive  (square of the radius of the ball)
     gamma   - positive, scalar or ND array compatible with the blocks of 'x'
                 [default: gamma=1]
     axis    - None or int, axis of block-wise processing [default: axis=None]
                  axis = None --> 'x' is processed as a single vector [DEFAULT] In this
                  case 'gamma' must be a scalar.
                  axis >=0   --> 'x' is processed block-wise along the specified axis
                  (0 -> rows, 1-> columns ect. in this case, 'gamma' and 'radius'
                  must be singletons along 'axis')
    """

    def __init__(
            self,
            center: np.ndarray,
            radius2: float,
            gamma: float or np.ndarray = 1,
            axis=None
            ):
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it is an array)" +
                " must be strictly positive"
            )
        if np.size(radius2) > 1 or radius2 < 0:
            raise Exception("'radius' must be a positive scalar")

        self.gamma = gamma
        self.r = radius2
        self.c = center
        self.axis = axis

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        scale = self.gamma

        center = self.c
        r = self.r   # square of the radius
        l2_norm = np.linalg.norm(x - center, ord=2, axis=self.axis)

        def polynom_phi(t):
            return t**3 - l2_norm * t**2 - (r + 2 * scale) * t + r * l2_norm

        def derive_polynom_phi(t):
            return 3 * t**2 - 2 * l2_norm * t - (r + 2 * scale)

        # starting point
        root_init = np.sqrt(r) / 2

        # Finding the root of the polynom with the Newton method
        root = newton_(
            polynom_phi, fp=derive_polynom_phi, x0=root_init, low=0, high=np.sqrt(r)
        )

        return center + (r - root**2) / (r - root**2 + 2 * scale) * (x - center)

    def __call__(self, x: np.ndarray) -> float:
        l2_norm2 = np.sum((x - self.c) ** 2, axis=self.axis)
        if np.all(l2_norm2 < self.r):
            return -np.sum(np.log(self.r - l2_norm2))
        return np.inf

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
