#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
% The function solves the 3rd-degree equation:
%
%        a * x^3 + b * x^2 + c * x + d = 0
%
% and returns the 3 roots in the output vectors y1, y2, y3.

% Version : 1.0 (06-18-2022)
% Author  : Mbaye Diongue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022
%
% This file is part of the codes provided at http://proximity-operator.net
%
% By downloading and/or using any of these files, you implicitly agree to 
% all the terms of the license CeCill-B (available online).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np


def solver_cubic(
        a: float or np.ndarray,
        b: float or np.ndarray,
        c: float or np.ndarray,
        d:float or np.ndarray
) -> float or np.ndarray:

    if np.any(a == 0):
        raise Exception("A cubic equation is expected: a cannot be null")
    if np.isscalar(a):
        a = np.array([a])
        b = np.array([b])
        c = np.array([c])
        d = np.array([d])

    # reduction to a depressed cubic (t^3 + p*t + q = 0)
    p = c / a - b**2 / (3 * a**2)
    q = (2 * b**3 / 27 - a * b * c / 3 + a**2 * d) / a**3

    # discriminant
    DD = (p / 3) ** 3 + (q / 2) ** 2

    # output vectors
    y1 = np.zeros_like(DD)  # the first root is always real
    y2 = np.zeros_like(DD, dtype=complex)
    y3 = np.zeros_like(DD, dtype=complex)

    # 1st case: 3 real unequal roots
    idx = DD < 0
    phi = np.arccos(-q[idx] / 2 / np.sqrt(np.abs(p[idx] ** 3 / 27)))
    tau = 2 * np.sqrt(np.abs(p[idx] / 3))
    y1[idx] = tau * np.cos(phi / 3)
    y2[idx] = -tau * np.cos((phi + np.pi) / 3)
    y3[idx] = -tau * np.cos((phi - np.pi) / 3)

    # 2nd case: 1 real root + 2 conjugate complex roots
    idx = DD > 0
    z1 = -q[idx] / 2
    z2 = np.sqrt(DD[idx])
    u = np.cbrt(z1 + z2)
    v = np.cbrt(z1 - z2)
    y1[idx] = u + v
    e1 = (-1 + 1j * np.sqrt(3)) / 2
    e2 = (-1 - 1j * np.sqrt(3)) / 2
    y2[idx] = u * e1 + v * e2
    y3[idx] = u * e2 + v * e1

    # 3rd case: 1 simple real root + 1 double real root
    idx = DD == 0
    y1[idx] = 3.0 * q[idx] / p[idx]
    y2[idx] = -1.5 * q[idx] / p[idx]
    y3[idx] = y2[idx]

    # correction to the original cubic
    t = b / (3 * a)
    y1 = y1 - t
    y2 = y2 - t
    y3 = y3 - t

    return [y1, y2, y3]
