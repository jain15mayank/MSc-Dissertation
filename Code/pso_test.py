# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 00:17:05 2019

@author: Mayank Jain
"""

import numpy as np
from pso_pyswarm import pso

def pso_objective(x):
    return np.sum(pow(x,2))

dim = 20
lb = -100*np.ones(dim)
ub = 100*np.ones(dim)
q_opt, f_opt = pso(pso_objective, lb, ub, swarmsize=40, omega=0.8, phip=2.0,
                   phig=2.0, maxiter=1000, minstep=1e-28, debug=True, processes=2)
