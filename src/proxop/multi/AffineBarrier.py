"""
Version : 1.0 ( 06-13-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class AffineBarrier:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the indicator of an affine barrier defined as:

                               / -log(b- a.T*x)    if u.T*x < b
                        f(x) =|
                              \   + inf            otherwise

    'gamma' is the scale factor


    As f is an indicator funtion, its proximity operator is the projection onto
    the affine barrier set.

     INPUTS
    ========
     x         - ND array
     a         - ND array with the same shape as 'x'
     b         - scalar or ND array compatible with the blocks of 'x'
     gamma     - positive, scalar or ND array compatible with the blocks of 'x'
             (default: gamma=1)
     axis     - None or int, axis of block-wise processing [default: axis=None]
              axis = None --> 'x' is processed as a single vector [DEFAULT] In this case
             'gamma' must be a scalar
              axis >=0   --> 'x' is processed block-wise along the specified axis
              (0 -> rows, 1-> columns ect. in this case, 'gamma' and 'epsilon'
              must be singletons along 'axis')

     =========
     Examples
     ==========


     >>> import numpy as np
     >>> from proxop import AffineBarrier
     >>>
     >>> x=np.array([1,2,3])
     >>> a= np.array([-1, 5, 3])
     >>> b= 3.5
     >>> AffineBarrier(x,b)(x)
     inf

    As the result below is infinite, that means 'x' does not belong to the affine set.

    Projection of x onto the affine set:

     >>> px= AffineBarrier(x,b).prox(x)
     >>> px
     >>> AffineBarrier(x,b)(px)
     -0.17973361

    As we could have suspected, the projection of 'x' belongs to the affine set.

     >>> x=np.arange(6)
     >>> x=x.reshape((2,3))
     >>> x
     array([[0, 1, 2],
           [3, 4, 5]])
     >>> a =np.ones
     >>> a[0,:]=2
     >>> a
     array([[2., 2., 2.],
            [1., 1., 1.]])
     >>> b=np.array([-1, 2, 4])

     Set 'axis=0' to process along the rows of the matrix 'x':

     >>> AffineBarrier(x,b, axis=0)(x)
     inf

     Projection of x onton the affine set:

     >>> px = AffineBarrier(x,b, axis=0).prox(x)
     >>> px
     array([[0.        , 0.05537521, 0.19926499],
            [0.        , 0.22150085, 0.49816248]])
     >>> AffineBarrier(x,b, axis=0)(px)
     array([-0.        , -0.05696748, -0.10495226])
    """

    def __init__(self,
                 a: np.ndarray,
                 b: np.ndarray,
                 gamma: np.ndarray = 1,
                 axis: int or None = None
                 ):
        if np.any(gamma <= 0):
            raise Exception(
                "'gamma' (or all of its components if it is an array)" +
                " must be strictly positive"
            )

        if (axis is not None) and (axis < 0):
            axis = None

        self.axis = axis
        self.gamma = gamma
        self.a = a
        self.b = b

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        scale = self.gamma
        a = self.a
        b = self.b
        sz = np.shape(x)
        sz = np.array(sz, dtype=int)
        sz[self.axis] = 1
        if np.size(scale) > 1:
            scale = np.reshape(scale, sz)
        if np.size(b) > 1:
            b = np.reshape(b, sz)

        scalar_prod = np.sum(self.a * x, axis=self.axis).reshape(sz)
        l2_a = np.sum(self.a ** 2, axis=self.axis).reshape(sz)
        return (
                x
                + (b - scalar_prod - np.sqrt((b - scalar_prod) ** 2 + 4 * scale * l2_a))
                / (2 * l2_a)
                * a
        )

    def __call__(self, x: np.ndarray) -> float:
        self._check(x)
        scalar_prod = np.sum(self.a * x, self.axis)
        b = self.b
        if np.size(b) != np.size(scalar_prod):
            raise ValueError("'b' must be compatible with the blocks of 'x'")
        b = np.reshape(b, np.shape(scalar_prod))
        res = np.zeros_like(1.0 * scalar_prod)
        mask = scalar_prod < self.b
        res[mask] = -np.log(b[mask] - scalar_prod[mask])
        res[np.logical_not(mask)] = np.inf
        return np.sum(res)

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
