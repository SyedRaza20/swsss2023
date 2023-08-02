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

    dx = 4.0 # kms

    # set x with 1 ghost cell on both sides:
    x = 100 + np.arange(-dx, 400 + 2*dx, dx)

    nPts = len(x)
    
    # the four important variables for adding the tides plus F 10.7 dependence:
    AmpDi = 10.0
    AmpSd = 5.0
    PhaseDi = np.pi/2
    PhaseSd = 3*np.pi/2
    
    # set default coefficients for the solver:
    a = np.zeros(nPts) - 1
    b = np.zeros(nPts) + 2
    c = np.zeros(nPts) - 1
    
    # lambda:
    lam = 80
    
    # the backgroung heat Q:
    Q = np.zeros(nPts)
    Q[(x>200)&(x<400)] = 0.4
    plt.plot(Q)
    dz = x[1] - x[0]
    dz2 = dz**2
    
    # making it time dependent:
    nDays = 27
    dt = 1 # in hours
    times = np.arange(0, nDays*24, dt) # in hours
    lon = 73.43
    
    # settign up the figure to plot:
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    
    # list to catch temperature
    temp = []
    
    # F10.7 dependence:
    f10_7 = 100 + (50/(24*365) * times) + (25 * np.sin(times/(27*24)) * 2 * np.pi)
    
    #plt.plot(f10_7)
    
    # do the for loop:
    for i, hour in enumerate(times):
        ut = hour % 24
        local_time = lon/15 + ut
        
        # the sun_heat:
        sun_heat = f10_7[i] * 0.4/100
        
        # the t_lower variable; adding the sin as the 10.7 dependence   
        t_lower = 200.0 + AmpDi  * np.sin((local_time/24) * 2*np.pi + PhaseDi) + AmpSd * np.sin((local_time/24) * 2*2*np.pi + PhaseSd)

    
        # the EUV heat array:
        Q_euv = np.zeros(nPts)
        fac = -np.cos(local_time * 2*np.pi / 24)
        if fac < 0:
            fac = 0
        Q_euv = np.zeros(nPts)
        Q_euv[(x>200)&(x<400)] = sun_heat * fac
        
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
        
    plt.contourf(times,x,temp,cmap="gray")
    plt.colorbar(label="Temperature (kelvin)")
    plt.xlabel("time (hours)")
    plt.ylabel("Altitude (km)")
    plt.title("Temperature above the longitude, " + str(lon))