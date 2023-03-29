# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:26:25 2023

@author: Uday Goel
"""

from openbb_terminal.sdk import openbb

import numpy as np

from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.stats import norm

annual_returns = (
    openbb.economy.index(
        ["^GSPC"], 
        start_date="1950-01-01", 
        column="Close"
    )
    .resample("A")
    .last()
    .pct_change()
    .dropna()
)
def norm_integral(f, mean, std):
    val, er = quad(
        lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std),
        mean - 3 * std,
        mean + 3 * std,
    )
    return -val
def get_kelly(data):
    solution = minimize_scalar(
        norm_integral, 
        args=(data["mean"], data["std"]),
        bounds=[0, 2],
        method="bounded"
    )
    return solution.x
annual_returns['f'] = return_params.apply(get_kelly, axis=1)
(
    annual_returns[["^GSPC"]]
    .assign(kelly=annual_returns["^GSPC"].mul(annual_returns.f.shift()))
    .dropna()
    .loc["1900":]
    .add(1)
    .cumprod()
    .sub(1)
    .plot(lw=2)
)