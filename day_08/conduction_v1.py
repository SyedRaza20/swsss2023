#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tridiagonal import solve_tridiagonal

# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

if __name__ == "__main__":

    dx = 0.25

    # set x with 1 ghost cell on both sides:
    x = np.arange(-dx, 10 + 2 * dx, dx)

    t_lower = 200.0
    #t_upper = 1000.0

    nPts = len(x)
    
    # lambda:
    lam = 100
    
    # the heat Q:
    Q = np.zeros(nPts)
    Q[(x>3)&(x<7)] = 100
    plt.plot(Q)
    dz = x[1] - x[0]
    dz2 = dz**2
    
    # make the new d:
    d = Q*(dz**2)/lam

    # set default coefficients for the solver:
    a = np.zeros(nPts) - 1
    b = np.zeros(nPts) + 2
    c = np.zeros(nPts) - 1
    
    # boundary conditions (bottom - fixed):
    a[0] = 0
    b[0] = 1
    c[0] = 0
    d[0] = t_lower

    # top - fixed:
    a[-1] = 1
    b[-1] = -1
    c[-1] = 0
    d[-1] = 0

    # Add a source term:
    
    # solve for Temperature:
    t = solve_tridiagonal(a, b, c, d)
    print(t)

    # plot:
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)

    ax.plot(x, t)
    ax.set_ylabel("Temperature")
    ax.set_xlabel("x")

    plotfile = 'conduction_v1.png'
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.close()
    
    
    
