# -*- coding: utf-8 -*-
"""
 Version : 1.0 (06-14-2022)
 
 Author  : Mbaye Diongue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Copyright (C) 2022

 This file is part of the codes provided at http://proximity-operator.net

 By downloading and/or using any of these files, you implicitly agree to 
 all the terms of the license CeCill-B (available online).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np


def prox_svd(x, gamma, prox_phi, hermitian=False):
    """
    This function computes the proximity operator of the matrix function:

                        f(X) = gamma * phi(s)     with   X = U' * diag(s) * V

     It is assumed that the matrices are stored along the dimensions 0 and 1.

      INPUTS
     ========
      x       - ND array
      gamma   - positive, scalar or ND array compatible with the size of 'x'
     prox_phi - function handle with two arguments at least
    """

    # spectral decomposition
    u, s, vh = np.linalg.svd(x, full_matrices=False, hermitian=hermitian)
    # prox computation
    g = np.reshape(prox_phi(s, gamma), np.shape(s))
    sz = np.shape(x)
    if len(sz) < 3:  # spectral reconstruction in the 2D case

        y = np.dot(u * g, vh)
        return y
    else:  # spectral reconstruction in the 4D case
        # szdim4= min( np.shape(u)[3], np.shape(vh)[3])
        y = np.matmul(u, g[..., None] * vh)
        return y
