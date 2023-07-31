#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to Space Weather Simulation Summer School Day 3

Today, we will be working with various file types, doing some simple data 
manipulation and data visualization

We will be using a lot of things that have been covered over the last two days 
with minor vairation.

Goal: Getting comfortable with reading and writing data, doing simple data 
manipulation, and visualizing data.

Task: Fill in the cells with the correct codes

@author: Peng Mun Siew
"""

#%% 
"""
This is a code cell that we can use to partition our data. (Similar to Matlab cell)
We hace the options to run the codes cell by cell by using the "Run current cell" button on the top.
"""
print ("Hello World")

#%%
"""
Creating a random numpy array
"""
# Importing the required packages
import numpy as np

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(10,5)
print(data_arr)

#%%
"""
TODO: Writing and reading numpy file
"""
# Save the data_arr variable into a .npy file
np.save("test_np_save.npy",data_arr)

# Load data from a .npy file
data_load = np.load("test_np_save.npy")

# Verify that the loaded data matches the initial data
print(data_load == data_arr)
print(np.equal(data_load, data_arr))

#%%
"""
TODO: Writing and reading numpy zip archive/file
"""
# Generate a second random array of dimension 8 by 1
data_arr2 = np.random.randn(8,1)
print(data_arr2)

# Save the data_arr and data_arr2 variables into a .npz file
np.savez("test_save.npz", data_arr, data_arr2)

# Load the numpy zip file
npzfile = np.load("test_save.npz")

# Verify that the loaded data matches the initial data
print(npzfile)
print("Variables within the zip file: ", sorted(npzfile.files))
#%%
"""
Error and exception
"""
# Exception handling, can be use with assertion as well
try:
    # Python will try to execute any code here, and if there is an exception 
    # skip to below 
    print(np.equal(data_arr,npzfile).all())
except:
    # Execute this code when there is an exception (unable to run code in try)
    print("The codes in try returned an error.")
    print(np.equal(data_arr,npzfile['arr_0']).all())
    
#%%
"""
TODO: Error solving 1
"""
# What is wrong with the following line? 
np.equal(data_arr,data_arr2)

#%%
"""
TODO: Error solving 2
"""
# What is wrong with the following line? 
np.equal(data_arr2,npzfile['data_arr2'])

#%%
"""
TODO: Error solving 3
"""
# What is wrong with the following line? 
numpy.equal(data_arr2,npzfile['arr_1'])

#%%
"""
Loading data from Matlab
"""

# Import required packages
import numpy as np
from scipy.io import loadmat

dir_density_Jb2008 = '/Users/syedraza/Desktop/michigansummerschool/swsss2023/day_03/JB2008/2002_JB2008_density.mat'

# Load Density Data
try:
    loaded_data = loadmat(dir_density_Jb2008)
    print (loaded_data)
except:
    print("File not found. Please check your directory")

# Uses key to extract our data of interest
JB2008_dens = loaded_data['densityData']

# The shape command now works
print(JB2008_dens.shape)

#%%
"""
Data visualization I

Let's visualize the density field for 400 KM at different time.
"""
# Import required packages
import matplotlib.pyplot as plt

# Before we can visualize our density data, we first need to generate the 
# discretization grid of the density data in 3D space. We will be using 
# np.linspace to create evenly sapce data between the limits.

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
nofLat_JB2008 = latitudes_JB2008.shape[0]

print(nofAlt_JB2008)
print(nofLst_JB2008)
print(nofLat_JB2008)

# We can also impose additional constratints such as forcing the values to be integers.
time_array_JB2008 = np.linspace(0,8759,5, dtype = int)

# For the dataset that we will be working with today, you will need to reshape 
# them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,
                                               nofAlt_JB2008,8760), order='F') # Fortran-like index order
#%%
"""
TODO: Plot the atmospheric density for 400 KM for the first time index in
      time_array_JB2008 (time_array_JB2008[0]).
"""

# Look for data that correspond to an altitude of 300 KM
alt = 300
hi = np.where(altitudes_JB2008==alt)
print("here what the altitude is: ", hi)

JB2008_dens_reshaped[:,:,hi,time_array_JB2008[0]].shape

#%%
"""
TODO: Plot the atmospheric density for 300 KM for all time indexes in
      time_array_JB2008
"""
import matplotlib.pyplot as plt
# make the figure and axes:
fig, axes = plt.subplots(5, figsize=(8,12))

# for loop for the following function:
for i in range(5):
    cs = axes[i].contourf(localSolarTimes_JB2008,latitudes_JB2008 , JB2008_dens_reshaped[:,:,hi,time_array_JB2008[i]].squeeze().T)
    axes[i].set_xlabel("The Local Solar Times")
    axes[i].set_ylabel("The Latitude")
    axes[i].set_title("Density at 300km")
    
    cbar = fig.colorbar(cs,ax = axes[i])
    cbar.ax.set_ylabel('Density')

#%%
"""
Assignment 1

Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identidy the time index that corresponds to  February 1st, 2002. 
# Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]
mean_density = np.zeros([nofAlt_JB2008])

mean_density = np.mean(dens_data_feb1, axis=(0,1))
plt.semilogy(altitudes_JB2008, mean_density)
plt.xlabel("Latitudes")
plt.ylabel("Mean Density")
plt.grid("on")

print('The dimension of the data are as followed (local solar time,latitude,altitude):', dens_data_feb1.shape)

#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density 
field at 310km

"""
# Import required packages
import h5py
loaded_data = h5py.File('/Users/syedraza/Desktop/michigansummerschool/swsss2023/day_03/TIEGCM/2002_TIEGCM_density.mat')

# This is a HDF5 dataset object, some similarity with a dictionary
print('Key within dataset:',list(loaded_data.keys()))
tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]

# We will be using the same time index as before.
time_array_tiegcm = time_array_JB2008

#%%
"""
TODO: Plot the atmospheric density for 310 KM for all time indexes in
      time_array_tiegcm
"""
tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,
                                               nofAlt_tiegcm,8760), order='F')
import matplotlib.pyplot as plt

# Look for data that correspond to an altitude of 400 KM
alt = 310
ht = np.where(altitudes_tiegcm==alt)
print(ht)

tiegcm_dens_reshaped[:,:,hi,time_array_tiegcm[0]]

# make the figure and axes:
fig, axes = plt.subplots(5, figsize=(8,12))

# for loop for the following function:
for i in range(5):
    cs = axes[i].contourf(localSolarTimes_tiegcm,latitudes_tiegcm , tiegcm_dens_reshaped[:,:,hi,time_array_tiegcm[i]].squeeze().T)
    axes[i].set_xlabel("The Local Solar Times")
    axes[i].set_ylabel("The Latitude")
    axes[i].set_title("Density at 310km")
    
    cbar = fig.colorbar(cs,ax = axes[i])
    cbar.ax.set_ylabel('Density')
    
print(JB2008_dens_reshaped.shape, tiegcm_dens_reshaped.shape)

#%%
"""
Assignment 1.5

Can you plot the mean density for each altitude at February 1st, 2002 for both 
models (JB2008 and TIE-GCM) on the same plot?
"""

# First identidy the time index that corresponds to  February 1st, 2002. 
# Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24


#%%
"""
Data Interpolation (1D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy import interpolate

# Let's first create some data for interpolation
x = np.arange(0, 10)
y = np.exp(-x/3.0)
# Make the 1D interpolation function:
interp_func_1D = interpolate.interp1d(x, y)

# Let's get some new interpolated points:
xnew = np.arange(0, 9, 0.1)
# find ynew based on interpolation function:
ynew = interp_func_1D(xnew)

print("The new x values: ", xnew)
print("The interpolated y values based on linear interpolation: ", ynew)

# Do 2D and 3D interpolation function:
interp_func_1D_cubic = interpolate.interp1d(x, y,kind='cubic')
interp_func_1D_quadratic = interpolate.interp1d(x, y,kind='quadratic')

# now interpolate
ycubic = interp_func_1D_quadratic(xnew)
yquadratic = interp_func_1D_cubic(xnew)

# plot the three interpolation functions: 
plt.subplots(1, figsize=(10, 6))
plt.plot(x, y, 'o', xnew, ynew, '*',xnew, ycubic, '--',xnew, yquadratic, '--',linewidth = 2)
plt.legend(['Inital Points','Interpolated line-linear','Interpolated line-cubic','Interpolated line-quadratic'], fontsize = 16)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.title('1D interpolation', fontsize=18)
plt.grid()
plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
#%%
"""
Data Interpolation (3D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolation on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)

# Generate thte interpolant (the interpolating function):
interpolated_function_1 = RegularGridInterpolator((x,y,z), sample_data)

# Say we are interested in the points [[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]]
pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
print('Using interpolation method:',interpolated_function_1(pts)) 
print('From true function:',function_1(pts[:,0],pts[:,1],pts[:,2]))

#%%
"""
Saving mat file

Now, let's us look at how to we can save our data into a mat file
"""
# Import required packages
from scipy.io import savemat

a = np.arange(20)
mdic = {"a": a, "label": "experiment"} # Using dictionary to store multiple variables
savemat("matlab_matrix.mat", mdic)

#%%
"""
Assignment 2 (a)

The two data that we have been working on today have different discretization 
grid.

Use 3D interpolation to evaluate the TIE-GCM density field at 400 KM on 
February 1st, 2002, with the discretized grid used for the JB2008 
((nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008).
"""
# Step 1: Generating the 3D interpolation function for the TIEGCM data:
tiegcm_interp_function = RegularGridInterpolator((localSolarTimes_tiegcm,latitudes_tiegcm, 
                                                   altitudes_tiegcm), tiegcm_dens_reshaped,
                                                 bounds_error=False, fill_value=None)

# Step 2: Evaluating the interpolant on the JB2008 data:
# (i) Create a 2D grid
mesh_grid_tiegcm = np.zeros((JB2008_dens_reshaped.shape[0], JB2008_dens_reshaped.shape[1]))

# (ii) make two for loops to interpolate the points on each of these points:
for lst_index in range(JB2008_dens_reshaped.shape[0]):
    for lat_index in range(JB2008_dens_reshaped.shape[1]):
        mesh_grid_tiegcm[lst_index, lat_index] = tiegcm_interp_function((localSolarTimes_JB2008[lst_index],
                                                                       latitudes_JB2008[lat_index], 400))[31*24]
print(mesh_grid_tiegcm)

# Step 3: Plot the contourf plot:
fig, axes = plt.subplots(2, figsize=(15,10))
cs1 = axes[0].contourf(localSolarTimes_JB2008, latitudes_JB2008, mesh_grid_tiegcm.T)
axes[0].set_xlabel("The Local Solar Times for JB2008")
axes[0].set_ylabel("The Latitude for JB2008")
axes[0].set_title("Density at 400km")
cbar = fig.colorbar(cs,ax = axes[0])
cbar.ax.set_ylabel('Density')

cs2 = axes[1].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,744].squeeze().T)
axes[1].set_xlabel("The Local Solar Times for JB2008")
axes[1].set_ylabel("The Latitude for JB2008")
axes[1].set_title("Density at 400km")
cbar = fig.colorbar(cs,ax = axes[1])
cbar.ax.set_ylabel('Density')
#%%
"""
Assignment 2 (b)

Now, let's find the difference between both density models and plot out this 
difference in a contour plot.
"""


fig, ax = plt.subplots(1, figsize=(20,8))
#plt.contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_array_JB2008[0]].squeeze().T)
difference_between_models=  mesh_grid_tiegcm.T - JB2008_dens_reshaped[:,:,hi,744].squeeze().T
cs = ax.contourf(localSolarTimes_JB2008, latitudes_JB2008, difference_between_models)
ax.set_title("Difference between the densities of the two models")
ax.set_ylabel("latitudes for JB2008 model")
ax.set_xlabel("Differences")

cbar = fig.colorbar(cs,ax = ax, )
cbar.ax.set_ylabel('Density')

#%%
"""
Assignment 2 (c)

In the scientific field, it is sometime more useful to plot the differences in 
terms of absolute percentage difference/error (APE). Let's plot the APE 
for this scenario.

APE = abs(tiegcm_dens-JB2008_dens)/tiegcm_dens
"""





