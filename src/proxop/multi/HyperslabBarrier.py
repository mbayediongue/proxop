"""
Version : 1.0 ( 06-13-2022).

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


class HyperslabBarrier:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the  indicator of a hyper-slab barrier function defined as:

                / -log( high - a.T*x)-log( a.T - low)   if low < u.T*x < high
         f(x) =|
               \   + inf                                otherwise


    'gamma' is the scale factor

     INPUTS
    ========
     x      - ND array
     a      - ND array with the same size as 'x'
     low    - scalar or ND array compatible with the blocks of 'x'[default: low  = -1]
     high   - scalar or ND array compatible with the blocks of 'x' [default: high = 1]
     gamma  - positive, scalar or ND array compatible with the blocks of 'x'
             [default: gamma = 1]
     axis   - None or int, axis of block-wise processing [default: axis=None]
                  axis = None --> 'x' is processed as a single vector [DEFAULT] In this
                  case 'gamma' must be a scalar.
                  axis >=0   --> 'x' is processed block-wise along the specified axis
                  (0 -> rows, 1-> columns ect. in this case, 'gamma' and 'epsilon'
                  must be singletons along 'axis')
    """

    def __init__(
            self,
            a: float or np.ndarray,
            low: float or np.ndarray = -1,
            high: float or np.ndarray = 1,
            axis: int or None = None,
            gamma: float or np.ndarray = 1
    ):
        if np.any(gamma <= 0):
            raise ValueError(
                "'gamma' (or all of its components if it is an array)" +
                " must be strictly positive"
            )
        if np.any(low > high):
            raise ValueError(" 'high' must be greater than 'low'")
        if np.size(low) <= 1:
            low = np.reshape(low, (-1))
            high = np.reshape(high, (-1))
        self.gamma = gamma
        self.a = a
        self.low = low
        self.high = high
        self.axis = axis

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        scale = self.gamma
        low = self.low
        high = self.high
        sz0 = np.shape(x)
        sz0 = np.array(sz0, dtype=int)
        sz0[self.axis] = 1
        if np.size(scale) > 1:
            scale = np.reshape(scale, sz0)

        sp = np.sum(self.a * x, self.axis)  # scalar product
        l2_a = np.sum(self.a ** 2, axis=self.axis)

        def polynom_phi(t):
            return (
                    t ** 3
                    - (low + high + sp) * t ** 2
                    + (low * high + sp * (low + high) - 2 * scale * l2_a) * t
                    - low * high * sp
                    + scale * (low + high) * l2_a
            )

        def derive_polynom_phi(t):
            return (
                    3 * t ** 2
                    - 2 * (low + high + sp) * t
                    + (low * high + sp * (low + high) - 2 * scale * l2_a)
            )

        # starting point
        root_init = (low + high) / 2

        # Find the root of the polynom with the Newton method
        root = newton_(
            polynom_phi, fp=derive_polynom_phi, x0=root_init, low=low, high=high
        )

        coef = np.reshape((root - sp) / l2_a, sz0)
        return x + coef * self.a

    def __call__(self, x: np.ndarray) -> float:
        self._check(x)
        scalar_prod = np.sum(self.a * x)
        if np.size(scalar_prod) <= 1:
            scalar_prod = np.reshape(scalar_prod, (-1))
        res = np.inf * np.ones_like(scalar_prod)
        mask = (scalar_prod < self.high) * (scalar_prod > self.low)
        res[mask] = -np.log(self.high[mask] - scalar_prod[mask]) - np.log(
            scalar_prod[mask] - self.low[mask]
        )

        return np.sum(self.gamma * res)

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
