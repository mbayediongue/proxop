"""
Created on Wed Jun  8 2022

@author: Mbaye Diongue

Implementation of Newton to give the root of a funtion in a given interval [low, high]
"""

import numpy as np


def newton_(
        f,
        fp,
        x0: float or np.ndarray,
        low: float or np.ndarray = -1. * np.inf,
        high: float or np.ndarray = np.inf,
        max_iter: int = 100
) -> float or np.ndarray:

    """Find an annulation point of f in [low, high] using Newton's method.

    Parameters
    ----------
    f : a function (callable), we assume it has a root in [a, b].
    fp : derivative of f
    x0 : starting point
    low : lower bound of the interval in which we look the root
    high : upper bound of the interval in which we look the root
    max_iter: maximum number of iteration

    Returns
    -------
    x: root of the function f  i.e f(x)=0
    """

    tol = 1e-8
    val = 1e-20  # to avoid the division by zero when the derivative is null
    stop = False
    i = 0
    x = x0
    while (i < max_iter) and (not stop):
        x_old = x

        x = x - f(x) / (val + fp(x))  # newton step
        x = np.minimum(np.maximum(low, x), high)  # range constraint

        err = np.linalg.norm(x - x_old)
        if err < tol:
            stop = True
        i += 1

    return x
