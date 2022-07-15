# -*- coding: utf-8 -*-
"""
Created 14-06-2022
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


def fun_svd(x: np.ndarray, gamma: float or np.ndarray, fun_phi, hermitian: bool = False) -> np.ndarray:
    """
    This function evaluates the function:

                        f(X) = gamma * phi(s)     with   X = U' * diag(s) * V

     It is assumed that the matrices are stored along the dimensions 0 and 1.

      INPUTS
     ========
      x       - ND array
      gamma   - positive, scalar or ND array compatible with the size of 'x'
      fun_phi - function handle with one argument at least
    """

    # spectral decomposition
    u, s, vh = np.linalg.svd(x, full_matrices=True, hermitian=hermitian)

    return gamma * fun_phi(s)
