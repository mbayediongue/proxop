"""
Version : 1.0 ( 06-22-2022).

DEPENDENCIES:
    - 'Max.py' located in the folder 'multi'

Author  : Mbaye Diongue

Copyright (C) 2022

This file is part of the codes provided at http://proximity-operator.net

By downloading and/or using any of these files, you implicitly agree to
all the terms of the license CeCill-B (available online).
"""


import numpy as np
from proxop.multi.Max import Max


class Linf(Max):
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the infinitive norm:

                        f(x) = gamma *tau* max( |x1|, ..., |xn|)

    INPUTS
    ========
     x         - ND array
     gamma     - positive, scalar or ND array compatible with the blocks of 'x'
                 [default: gamma=1], 'gamma' is the scale factor
     axis    - None or int, axis of block-wise processing [default: axis=None]
                  axis = None --> 'x' is processed as a single vector [DEFAULT] In this
                  case 'gamma' must be a scalar.
                  axis >=0   --> 'x' is processed block-wise along the specified axis
                  (0 -> rows, 1-> columns ect. In this case, 'gamma' must be singleton
                  along 'axis'.
    """

    def __init__(self, gamma=1, axis=None):
        super().__init__(gamma=gamma, axis=axis)

    def prox(self, x: np.ndarray) -> np.ndarray:
        scale = self.gamma
        abs_x = np.abs(x)
        return np.sign(x) * Max(gamma=scale, axis=self.axis).prox(abs_x)

    def __call__(self, x: np.ndarray) -> float:
        Max._check(self, x)
        l1 = np.max(np.abs(x), self.axis)
        g = self.gamma
        if np.size(g) <= 1:
            return np.sum(g * l1)
        g = np.reshape(g, l1.shape)
        return np.sum(g * l1)
