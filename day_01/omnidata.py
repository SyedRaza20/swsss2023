#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:51:33 2023

@author: syedraza
@email: sar0033@uah.edu

This program plots auroral electroject data for the author's birthday
"""

# Performing the needed imports 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from swmfpy.web import get_omni_data

# Getting the start and the end date
start_data = datetime(1998,10,30)
end_data = datetime(1998,10,31)

# Getting the data needed:
data = get_omni_data(start_data, end_data)

# plotting the data["times"] and data["al"]
plt.figure(figsize=(10,6))
plt.plot(data["times"], data["al"])
plt.xlabel("$times$")
plt.ylabel("$al$")
plt.title("OMNI auroral data for Syed's Birthday")
plt.show()