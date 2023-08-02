#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:14:00 2023

"""

__author__ = "Syed Raza"
__email__ = "sar0033@uah.edu"

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
    
    # set default coefficients for the solver:
    a = np.zeros(nPts) - 1
    b = np.zeros(nPts) + 2
    c = np.zeros(nPts) - 1
    
    # lambda:
    lam = 10
    
    # the backgroung heat Q:
    sun_heat = 100
    Q = np.zeros(nPts)
    Q[(x>3)&(x<7)] = sun_heat
    plt.plot(Q)
    dz = x[1] - x[0]
    dz2 = dz**2
    
    # making it time dependent:
    local_time = 0
    nDays = 3
    dt = 1 # in hours
    times = np.arange(0, nDays*24, dt) # in hours
    lon = 73.43
    print(times)
    
    # settign up the figure to plot:
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    
    # list to catch temperature
    temp = []
    
    # do the for loop:
    for hour in times:
        ut = hour % 24
        local_time = lon/15 + ut
    
        # the EUV heat array:
        Q_euv = np.zeros(nPts)
        fac = -np.cos(local_time * 2*np.pi / 24)
        if fac < 0:
            fac = 0
        Q_euv = np.zeros(nPts)
        Q_euv[(x>3)&(x<7)] = sun_heat * fac
        
        # make the new d:
        d = (Q + Q_euv)*(dz**2)/lam
        
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
        temp.append(t)
        
        """
        ax.plot(x,t)
        ax.set_ylabel("Temperature")
        ax.set_xlabel("Altitude")
        ax.set_title("Temperature vs Altitude")
        
        plotfile = 'conduction_v1b.png'
        print('writing : ',plotfile)    
        fig.savefig(plotfile)
        """
        
    temp = np.array(temp).T
    
    # the altitude:
    alt = 100 + 40*x
        
    plt.contourf(times,alt,temp,cmap="gray")
    plt.colorbar(label="Temperature (kelvin)")
    plt.xlabel("time (hours)")
    plt.ylabel("Altitude (km)")
    plt.title("Temperature above the longitude, " + str(lon))