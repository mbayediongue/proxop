"""
Version : 1.0 (06-18-2022).

Author  : Mbaye DIONGUE

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""

import numpy as np


class ConicL0:
    r"""Compute the proximity operator and the evaluation of the gamma*f.

    where the function f is defined as:

                   f(x,d) = gamma \ell_0(x) + \iota_{S}(x,d)

    with S = {x \in C^N, d \in C^N, s.t. (\forall n \in {1,...,N})
              \exist delta_n \in [-Delta_n,Delta_n] s.t. d_n = \delta_n x_n
              }
    where \Delta \in [0,+\infty)^N

    When the inputs '(x,d)' are arrays, the outputs '(p,q)' are computed element-wise.

     INPUTS
    ========
     x     - ND array complex valued
     d     - ND array complex valued with the same size as 'x'
     gamma - positive, scalar
     Delta - real positive,  scalar or ND array with the same size as 'x'

    Note: When calling the function (and not the proximity operator) the result
    is computed element-wise SUM. So the command >>>ConicL0(Delta=2, d=2*x)(x) will
    return a scalar even if x is a vector:
    >>> x=np.array([ 1+1j, 4+2J])
    >>> x
    array([1.+1.j, 4.+2.j])
    >>> ConicL0(Delta=3, d=2*x)(x)
    2

    But as expected, >>> ConicL0(Delta=3, d=2*x).prox(x)
    will return two vectors with the same size as x:

    >>> px=ConicL0(Delta=3, d=2*x).prox(x)
    [array([1.+1.j, 4.+2.j]), array([2.+2.j, 8.+4.j])]
    """

    def __init__(
        self,
        Delta: float or np.ndarray,
        d: np.ndarray,
        gamma: float or np.ndarray = 1.0
    ):

        if np.any(gamma <= 0):
            raise Exception("'gamma'(or all of its elements) must be strictly positive")
        if np.any(Delta < 0):
            raise Exception("'lamb'(or all of its elements) must be positive or null")

        self.Delta = Delta
        self.gamma = gamma
        self.d = d

    def prox(self, x: np.ndarray) -> [np.ndarray, np.ndarray]:
        self._check(x)
        self._check_size(x)
        gamma = self.gamma
        Delta = self.Delta
        d = self.d

        prox_x = np.zeros_like(1.0 * x)
        prox_q = np.zeros_like(1.0 * x)

        z = np.real(x * np.conj(d))
        abs_d2 = np.abs(d) ** 2
        abs_x2 = np.abs(x) ** 2

        eta = np.sqrt(((abs_d2 - abs_x2)) ** 2 + 4 * z**2)

        delta_hat = Delta * np.ones(np.shape(x))
        mask = z != 0
        delta_hat1 = (eta + abs_d2 - abs_x2) / (2 * np.abs(z))
        delta_hat[mask] = (np.minimum(delta_hat1, Delta) * np.sign(z))[mask]
        delta_hat[(z == 0) * (abs_x2 >= abs_d2)] = 0

        seuil_mod = (np.abs(delta_hat * x - d) ** 2) / (1 + delta_hat**2) + 2 * gamma
        ind_seuil = abs_x2 + abs_d2 >= seuil_mod

        upmod = (x + delta_hat * d) / (1 + delta_hat**2)

        prox_x[ind_seuil] = upmod[ind_seuil]
        prox_q[ind_seuil] = upmod[ind_seuil] * delta_hat[ind_seuil]
        return [prox_x, prox_q]

    def __call__(self, x: np.ndarray) -> float:
        self._check_size(x)
        d = self.d
        Delta = self.Delta
        if np.size(x) <= 1:
            x = np.reshape(x, (-1))
        if np.size(d) <= 1:
            d = np.array([d])
        tol = 1e-15
        # check the constraints
        # x and d does not have the same arg
        if abs(np.sum(np.imag(x * np.conj(d)))) > tol:
            return np.inf

        # x and d have the same arg
        mask_x = x == 0
        mask_d = d == 0
        if np.any(mask_x != mask_d):
            return np.inf
        # evaluate the l0 function
        fun_x = np.sum(self.gamma * (x != 0))

        if np.size(mask_x) == np.size(x):
            return fun_x

        mask = x != 0
        # delta not in [-Delta;Delta]
        if np.any(np.abs(x[mask]) / np.abs(d[mask]) > Delta):
            return np.inf

        return fun_x

    def _check(self, x):
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != np.size(x)):
            raise Exception("'gamma' must be either scalar or the same size as 'x'")

    def _check_size(self, x):
        sz_x = np.size(x)
        if np.size(self.d) > 1 and (np.size(self.d) != sz_x):
            raise Exception("'d' must have the same size as 'x'")
        if (np.size(self.Delta) > 1) and (np.size(self.Delta) != sz_x):
            raise Exception("'Delta' must be either scalar or the same size as 'x'")
        if (np.size(self.gamma) > 1) and (np.size(self.gamma) != sz_x):
            raise Exception("'gamma' must be either scalar or the same size as 'x'")
