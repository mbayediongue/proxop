"""
Version : 1.0 ( 06-22-2022).

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np
import numpy.fft as fft


class L2DownConv:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:

                        f(x) = || S_d*H*x - y||_2 ^2

    Where S_d is the down sampling operator with factor d>0 and H is a convolution
    operator that is described by the Fourier Kernel Lamb
    [i.e. Hx = ifftn( Lamb*fftn(x) )]

     INPUTS
    ========
     x         - ND array
     Lamb      - ND array with the same size as 'x' ( Fourier kernel of the
                convolution operator H)
     d         - int or vector of positive integers (down sampling in each dimension of x)
     gamma     - positive scalar
     y         - ND array with size : size(x)/d
    """

    def __init__(
            self,
            lamb: np.ndarray,
            d: int or np.ndarray,
            y: np.ndarray,
            gamma: float = 1.0
    ):
        if np.size(gamma) > 1 or gamma <= 0:
            raise Exception("'gamma' must be a strictly positive scalar")
        d = np.array(d, dtype=int)
        self.gamma = gamma
        self.lamb = lamb
        self.d = d
        self.y = y

        # pre-Computations
        sz = np.shape(lamb)
        mask_ = np.zeros(np.size(lamb), dtype=bool)

        if np.size(d) <= 1:
            mask_[np.arange(1, sz[0], d)] = True
        else:
            for ii in range(np.size(d)):
                sample = np.zeros(sz[ii], dtype=bool)
                sample[np.arange(0, sz[ii], d[ii])] = True
                mask_[ii * sz[ii]:(ii + 1) * sz[ii]] = sample
        self.mask_ = mask_.reshape(np.shape(lamb))

    def prox(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        scale = self.gamma
        y = self.y
        d = self.d
        lamb = self.lamb
        y_up = np.zeros(np.shape(x))
        y_up[self.mask_] = np.reshape(y, (-1))[0:np.sum(1 * self.mask_)]
        sz_ = np.array(np.shape(x))
        sz_patch = np.array(sz_ / d, dtype=int)

        def expand_mat(mat: np.ndarray, size_patch: np.ndarray) -> np.ndarray:
            res: np.ndarray = np.zeros(size_patch)
            sz_2 = []
            for k in range(len(size_patch)):
                sz_2.append(size_patch[k])
            sz_2.append(-1)
            sz_2 = tuple(sz_2)
            tmp = np.reshape(mat, sz_2)
            for n in range(np.shape(tmp)[-1]):
                res = res + tmp[..., n]
            return res

        h2_h2t = expand_mat(np.abs(lamb) ** 2, sz_patch)

        def rep_mat(xx: np.ndarray):
            return np.tile(xx, d)

        # Prox computation
        fft_reverse = np.conj(lamb) * np.fft.fftn(y_up) + np.fft.fftn(x / scale)

        prox_x = fft.ifftn(
            scale
            * (
                    fft_reverse
                    - np.conj(lamb)
                    * rep_mat(expand_mat(lamb * fft_reverse,
                                         sz_patch) / (np.size(d) / scale +
                                                      h2_h2t)).reshape(np.shape(x))
            )
        )

        return prox_x.real

    def __call__(self, x: np.ndarray) -> float:
        hx = fft.fftn(self.lamb * np.fft.ifftn(x))
        hx = hx.real  # the real part of the Fourier transform
        yy = np.reshape(self.y, (-1))[0:np.sum(1 * self.mask_)]
        res = self.gamma / 2 * np.sum(hx[self.mask_] - yy) ** 2
        return res

    def _check(self, x):
        if np.any(np.array(np.shape(self.y)) != np.array(np.shape(x)) / self.d):
            raise Exception("'y' must be an ND array with shape: np.shape(x)/d")
        if np.size(self.lamb) != np.size(x):
            raise Exception("'Lamb' must be an ND array with size as 'x'")
