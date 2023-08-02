#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:41:10 2023

@author: holtorf
"""

import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

# Use Euler's method with different stepsizes to solve the IVP:
# dx/dt = -2*x, with x(0) = 3 over the time-horizon [0,2]

# Compare the numerical approximation of the IVP solution to its analytical
# solution by plotting both solutions in the same figure. 

def func(x):
    """
    This function just returns the x(t) based on the problem.
    """
    return -2*x

def analytical_x(t):
    """

    Parameters
    ----------
    t : np.array (type)
        This is the time variable "t" that x is dependent on 

    Returns
    -------
    x_t : np.array (type)
        This is the analytical solution x_t 
    """
    return 3*np.exp(-2*t)

def explicit_euler(h):
    """
    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # make the x array with the required step-size:
    x = np.zeros(len(np.arange(0,2,h)))
    
    # initial condition:
    x_0 = 3
    x[0] = x_0
    
    for i in range(1, len(x)):
        x[i] = x[i-1] + h*func(x[i-1])
    
    return x

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# THE MAIN BLOCK
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    # make a random t:
    h = 0.01
    t = np.arange(0,2,h)

    fig, ax = plt.subplots()
    # get the analytical solution:
    ana = analytical_x(t)
    eu = explicit_euler(h)
    print("I am here: ", eu)
    
    
    ax.plot(t, ana, label="analytical solution", color="blue")
    ax.plot(t, eu, label="explicit euler method", color="red")
    
    ax.set_xlabel("time")
    ax.set_ylabel("Solution")
    ax.legend()
    
    
    
    