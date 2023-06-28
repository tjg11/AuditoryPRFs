### Author: Taylor Garrison
### Description: Designed to convolve a hemodynamic response function (HRF) with a stimulus
### created from the stimulus_creation.py script in this respository. Convolved stimulus
### used in following analysis. Saved out as a binary file.

import scipy
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import pickle
from os import path as op

# Define the HRF using parametric function

dt = 0.01
delta = 2.25 # this is the one that varies the most across subjects and brain location
tau = 1.2 # known to also change across the cortex
n = 3

th = np.arange(0, 30, dt)

t_delta = th - delta # shift time vector for lag
h = (((t_delta / tau) ** (n - 1)) * np.exp(-t_delta / tau)) / (tau * np.math.factorial(n - 1))
h[th < delta] = 0 # remove values prior to delta lag

plt.plot(th, h)
plt.show()

# Open binarized data file from stimulus_creation.py
f_name = op.join("sub-EBxGxCCx1986", "binary_sub-EBxGxCCx1986_1.pickle")
with open (f_name, "rb") as f:
    S = pickle.load(f)

# Print shape and plot data for sanity check
print(f"Shape of subject data: {S.shape}")
plt.imshow(S)
plt.ylim(0, 160)
plt.show()

C = np.apply_along_axis(
    lambda x: scipy.signal.convolve(x, h, mode = "same"), 
    axis = 0, arr = S)

print(C.shape)
img_name = "convolved_sub-EBxGxCCx1986_1.pickle"
img_name = op.join("sub-EBxGxCCx1986", img_name)
with open(img_name, "wb") as f:
    pickle.dump(C, f)