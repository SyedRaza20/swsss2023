#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:59:14 2023

A 3D plot script for spherical coordinates.
"""

__author__ = "Syed Raza"
__email__ = "sar0033@uah.edu"

# import statements:
import numpy as np
import matplotlib.pyplot as plt

# make a function that converts spherical coordinates to cartesian coordinates
def spherical_cartesian(r, theta, phi):
    """
    This is a function that converts spherical coordinates to cartesian 
    coordinates.
    
    NOTE: This function assumes that the input arguments are in radians and NOT
    degrees
    
    Args:
        --> phi: this is the angle between the z-axis and r
        --> theta: this is the angle between the x-axis and the projection of r 
        onto the xy-plane
        --> r is the radial component of the coordinate system
        
    Returns:
        --> x = r.sin(phi).cos(theta)
        --> y = r.sin(phi).sin(theta)
        --> z = r.cos(phi)
        
        AKA; the cartesian coordinates. This function will return a tuple of 
        the form (x,y,z).
        
    Testing examples:
    """
    
    # getting the x value:
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    
    # returning the tuple
    return (x,y,z)

# unit testing: This should do nothing if the code works well:
assert spherical_cartesian(1, 2*np.pi, 2*np.pi) > (0-1.e-5 ,0-1.e-5 ,1-1.e-5) and spherical_cartesian(1, 2*np.pi, 2*np.pi) < (0+1.e-5 ,0+1.e-5 ,1+1.e-5), "It doesn't work!!"
assert spherical_cartesian(1, np.pi, np.pi) > (0-1.e-5 ,0-1.e-5 ,-1-1.e-5) and spherical_cartesian(1, np.pi, np.pi) < (0+1.e-5 ,0+1.e-5 ,-1+1.e-5), "It doesn't work!!"
assert spherical_cartesian(1, 2*np.pi, 2*np.pi) > (0-1.e-5 ,0-1.e-5 ,1-1.e-5) and spherical_cartesian(1, 2*np.pi, 2*np.pi) < (0+1.e-5 ,0+1.e-5 ,1+1.e-5), "It doesn't work!!"
assert spherical_cartesian(1, -np.pi, -2*np.pi) > (0-1.e-5 ,0-1.e-5 ,-1-1.e-5) and spherical_cartesian(1, -np.pi, -2*np.pi) < (0+1.e-5 ,0+1.e-5 ,-1+1.e-5), "It doesn't work!!"
assert spherical_cartesian(1, -2*np.pi, -np.pi) > (0-1.e-5 ,0-1.e-5 ,1-1.e-5) and spherical_cartesian(1, -2*np.pi, -np.pi) < (0+1.e-5 ,0+1.e-5 ,1+1.e-5), "It doesn't work!!"

# plotting in the 3D plane:
fig = plt.figure()
axes = plt.axes(projection="3d")

# making the coordinate arrays for their standard values:
r = np.linspace(0,1)
theta = np.linspace(0, 2*np.pi)
phi = np.linspace(0, 2*np.pi)

x, y, z = spherical_cartesian(r, theta, phi)
axes.plot(x, y, z)