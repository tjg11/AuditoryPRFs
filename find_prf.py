### Author: Taylor Garrison
### Description: Designed to utilize outputs from stimulus_creation.py and convolve_hrf.py to
### determine and visualize population receptive fields (PRFs) based on parametized predicted
### time course.

from os import path as op
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import pickle
import scipy
import glob
import time
import pathlib
import os

print("\n+-------------Script Starts Here-------------+\n")

# Print sanity check statements and plots?
sanity_check = True
# sanity_check = False
plot_timecourses = True
# plot_timecourse = False

# Pick a subject
subject_id = "sub-EBxGxCCx1986"

# Load one data file
# For analyzing specific files

base_path = os.path.dirname(os.path.abspath(__file__))
img_name = op.join(base_path, subject_id, ("convolved_" + subject_id + "_1.pickle"))
with open(img_name, "rb") as f:
    loaded_record = pickle.load(f)

 # Sanity check the loaded file
if sanity_check:
    print("Total length of run should be 416 seconds and total rows per second should be 100. Shape should be 41600, 9.\n")
    print(f"Shape is: {loaded_record.shape}.")
    print(f"Total length of run: {loaded_record.shape[0] / 100} seconds.")
    print(f"Total rows per second: {loaded_record.shape[0] / 416} rows.")
   
# Generate gaussian function
def generate_gaussian(x, b, c):
    return np.exp(-(((x - b) ** 2) / (2 * c ** 2)))

# Pad beginning of stimulus with 8 seconds of zeros (so 800 rows).
# Padding parameters = (before, after)
x_pad = (800, 600)
y_pad = (0, 0)

padded_stim = np.pad(loaded_record, (x_pad, y_pad), mode='constant')

# Sanity check the padded file
if sanity_check:
    print("Total length of run should be 430 seconds and total rows per second should be 100. Shape should be 43000, 9.\n")
    print(f"Shape is: {padded_stim.shape}.")
    print(f"Total length of run: {padded_stim.shape[0] / 100} seconds.")
    print(f"Total rows per second: {padded_stim.shape[0] / 430} rows.")

# Interpolate to TR sampling rate (2 seconds).

total_sec = padded_stim.shape[0] / 100 # total time
dt = 0.01 # sampling rate
ts = np.arange(0, total_sec, dt) # time vector

tr_func = scipy.interpolate.interp1d(ts, padded_stim, axis = 0)
tr = np.arange(0, ts[-1], 2)
c_tr = tr_func(tr)

if sanity_check:
    print("Shape should be 215,9. There should be 215 trials.")
    print(f"Shape is {c_tr.shape}.")
    print(f"There are {c_tr.shape[0]} trials.")

# Load brain data

brain_path = op.join(base_path, "..", "..", "AMPB", "data",subject_id, "ses-02", "func", (subject_id + "_ses-02_task-ampb_run-1_bold.nii.gz"))
bold_img = nib.load(brain_path)
bold_data = bold_img.get_fdata()
print(f"Shape of brain data is {bold_data.shape}.")

# Select a voxel

x_coord = 50
y_coord = 22
z_coord = 21
voxel = bold_data[x_coord, y_coord, z_coord, :]
norm_voxel = (voxel - np.amin(voxel)) / (np.amax(voxel) - np.amin(voxel))

# Generate multiple gaussians and find best fit

x = np.linspace(-30, 30, 9)
 
mus    = np.arange(-41, 40, 2)
sigmas = np.arange(-41, 40, 2)

# mus    = np.arange(-21, 20, 2)
# sigmas = np.arange(-21, 20, 2)
mus, sigmas = np.meshgrid(mus, sigmas)


error_matrix = np.zeros(mus.shape).flatten()
iterable = zip(mus.flatten(), sigmas.flatten())

for i, params in enumerate(iterable):
    mu, sigma = params # current parameters
    # print(params)
    try:

        func = generate_gaussian(x, mu, sigma)
        pred_t = np.matmul(c_tr, func)
        corr = pearsonr(voxel, pred_t).statistic
        # print(corr)
        error_matrix[i] = -corr
    except Exception as error:
        print(error)
error_matrix = error_matrix.reshape(mus.shape)

# Find where error was at a minimum
print(f"Min error is: {np.nanmin(error_matrix)}.\n")

min_x, min_y = np.where(error_matrix == np.nanmin(error_matrix))

print(f"Min error is at {min_x[0]}, {min_y[0]}, which is a mu of {mus[min_x[0]][min_y[0]]} and a sigma of {sigmas[min_x[0]][min_y[0]]}.\n")

seed_mu    = mus[min_x[0]][min_y[0]]
seed_sigma = sigmas[min_x[0]][min_y[0]]

# Model function: create stim space and create gaussian model
stim_space = np.linspace(-30, 30, 9)
print(stim_space)

# Takes mu and theta as array and stim_space as array
def model_function(params, stim_space):
     return np.exp(-(((stim_space - params[0]) ** 2) / (2 * params[1] ** 2)))

# Prediction function: multiply convolved stimulus and model
def prediction_function(model, convolved_stim):
    return np.matmul(convolved_stim, model )

# Error function: get mode, prediction, and calculate error between prediction and real data
def error_function(params, stim_space, real_data, convolved_stim):
    # Get model
    model = model_function(params, stim_space)

    # Get y hat
    pred  = prediction_function(model, convolved_stim) # something to get y_pred
    
    # Normalize real data and pred data - move normalization of real data
    if np.amax(pred) != np.amin(pred):
        pred = (pred - np.amin(pred)) / (np.amax(pred) - np.amin(pred))
    
    # Calculate MSE
    error = mean_squared_error(pred, real_data) # or just math
    
    return error

# based on best mu and sigma get grouping of values around each (the best_mu and best_si values can be found wihtout manually entering them, I just need 
# to write the code for it)
tic = time.time()

best_mu = 10
best_si = 0.1

best_mus = np.arange(best_mu - 3, best_mu + 3, 1)
best_sis = np.arange(best_si - 3, best_mu + 3, 1)

best_mus, best_sis = np.meshgrid(best_mus, best_sis)
iterable = zip(best_mus.flatten(), best_sis.flatten())

# initialize dictionary and find best seed among grid
grid_results = dict()
grid_results['voxel'] = [x_coord, y_coord, z_coord]
grid_results['error'] = float('inf')
for i, seed in enumerate(iterable):
    try:
        results = minimize(error_function, seed, (stim_space, norm_voxel, c_tr), method='Nelder-Mead')
        if results.fun < grid_results['error']:
            grid_results['error'] = results.fun
            grid_results['best_mu'] = results.x[0]
            grid_results['best_sigma'] = results.x[1]
    except:
        print(f"Could not calculate for {seed}.")
        print("")

toc = time.time()
    
    
# print results of grid search
print(f"Results for voxel: {grid_results['voxel']} ")
print(f"Min error: {grid_results['error']}")
print(f"Best mu: {grid_results['best_mu']}")
print(f"Nest sigma: {grid_results['best_sigma']}")
print(f"Time taken: {toc - tic} seconds\n")

# perform minimization on best seed
best_seed = [grid_results['best_mu'], grid_results['best_sigma']]
the_results = minimize(error_function, best_seed, (stim_space, norm_voxel, c_tr), method = "Nelder-Mead") # not in for loop
print(f"Best mu and sigma: {the_results.x}")
print(f"Most minimized error: {the_results.fun}")




