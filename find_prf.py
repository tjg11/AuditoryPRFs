# Author: Taylor Garrison
# Description: Designed to utilize outputs from stimulus_creation.py and
# convolve_hrf.py to determine and visualize population receptive fields (PRFs)
# based on parametized predicted time course.

from os import path as op
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import pickle
import scipy
import time
import os

print("\n+-------------Script Starts Here-------------+\n")

# Print sanity check statements and plots?
sanity_check = True
# sanity_check = False
plot_timecourses = True
# plot_timecourse = False
print_status = False

# Pick a subject
subject_id = "sub-EBxGxCCx1986"

# Load one data file
# For analyzing specific files

base_path = os.path.dirname(os.path.abspath(__file__))
img_name = op.join(base_path,
                   subject_id,
                   ("convolved_" + subject_id + "_1.pickle"))
with open(img_name, "rb") as f:
    loaded_record = pickle.load(f)

# Sanity check the loaded file
if sanity_check:
    print(
        f"""Total length of run should be 416 seconds and total rows
        per second should be 100. Shape should be 41600, 9.
        Shape is: {loaded_record.shape}.
        Total length of run: {loaded_record.shape[0] / 100} seconds.
        Total rows per second: {loaded_record.shape[0] / 416} rows."""
    )


# Generate gaussian function
def generate_gaussian(x, b, c):
    return np.exp(-(((x - b) ** 2) / (2 * c ** 2)))


# Takes mu and theta as array and stim_space as array
def model_function(params, stim_space):
    return np.exp(-(((stim_space - params[0]) ** 2) / (2 * params[1] ** 2)))


# Prediction function: multiply convolved stimulus and model
def prediction_function(model, convolved_stim):
    return np.matmul(convolved_stim, model)


# Error function: get mode, prediction, and calculate error between
# prediction and real data
def error_function(params, stim_space, real_data, convolved_stim):
    # Get model
    model = model_function(params, stim_space)

    # Get y hat
    pred = prediction_function(model, convolved_stim)

    # Normalize real data and pred data - move normalization of real data
    if np.amax(pred) != np.amin(pred):
        pred = (pred - np.amin(pred)) / (np.amax(pred) - np.amin(pred))

    # Calculate MSE
    error = mean_squared_error(pred, real_data)

    return error


# Pad beginning of stimulus with 8 seconds of zeros (so 800 rows).
# Padding parameters = (before, after)
x_pad = (800, 600)
y_pad = (0, 0)

padded_stim = np.pad(loaded_record, (x_pad, y_pad), mode='constant')

# Sanity check the padded file
if sanity_check:
    print(f"""Total length of run should be 430 seconds and total rows per
        second should be 100. Shape should be 43000, 9.
        Shape is: {padded_stim.shape}
        Total length of run: {padded_stim.shape[0] / 100} seconds
        Total rows per second: {padded_stim.shape[0] / 430} rows."""
          )

# Interpolate to TR sampling rate (2 seconds).

total_sec = padded_stim.shape[0] / 100
dt = 0.01  # sampling rate
ts = np.arange(0, total_sec, dt)  # time vector

tr_func = scipy.interpolate.interp1d(ts, padded_stim, axis=0)
tr = np.arange(0, ts[-1], 2)
c_tr = tr_func(tr)

if sanity_check:
    print("Shape should be 215,9. There should be 215 trials.")
    print(f"Shape is {c_tr.shape}.")
    print(f"There are {c_tr.shape[0]} trials.")

# Load brain data

brain_path = op.join(base_path,
                     "..",
                     "..",
                     "AMPB",
                     "data",
                     subject_id,
                     "ses-02",
                     "func",
                     (subject_id + "_ses-02_task-ampb_run-1_bold.nii.gz"))
bold_img = nib.load(brain_path)
bold_data = bold_img.get_fdata()
print(f"Shape of brain data is {bold_data.shape}.")

# Generate voxel coordinate grid
x_start = 40
x_end = 50

y_start = 30
y_end = 40

z_start = 21
z_end = 22

x_coords = np.arange(x_start, x_end)
y_coords = np.arange(y_start, y_end)
z_coords = np.arange(z_start, z_end)

xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords,  indexing='ij')

grid_coords = np.vstack([xv.flatten(), yv.flatten(), zv.flatten()]).T

total_voxel = grid_coords.shape[0]
done_voxel = 0

# Generate mu / sigma grid

stim_space = np.linspace(-30, 30, 9)


# mus    = np.arange(-21, 20, 2)
# sigmas = np.arange(-21, 20, 2)


all_results = []
tic = time.time()
for coord in grid_coords:  # for each coordinate in voxel grid

    # Select a voxel - this needs to be done in loop
    x, y, z = coord
    voxel = bold_data[x, y, z, :]
    norm_voxel = (voxel - np.amin(voxel)) / (np.amax(voxel) - np.amin(voxel))

    print(f"\n-----Starting for {x, y, z} voxel------\n")

# Generate multiple gaussians and find best fit - this also needs to be done in
# loop since voxel will be different

    mus = np.arange(-41, 40, 2)
    sigmas = np.arange(-41, 40, 2)
    mus, sigmas = np.meshgrid(mus, sigmas)

    error_matrix = np.zeros(mus.shape).flatten()
    all_params = zip(mus.flatten(), sigmas.flatten())

    for i, params in enumerate(all_params):
        mu, sigma = params  # current parameters
        func = generate_gaussian(stim_space, mu, sigma)
        pred_t = np.matmul(c_tr, func)
        corr = pearsonr(voxel, pred_t).statistic
        error_matrix[i] = -corr

    error_matrix = error_matrix.reshape(mus.shape)

    # Find where error was at a minimum
    if print_status:
        print(f"Min error is: {np.nanmin(error_matrix)}.\n")

    min_x, min_y = np.where(error_matrix == np.nanmin(error_matrix))
    if print_status:
        print(
            f"""Min error is at {min_x[0]}, {min_y[0]},
            which is a mu of {mus[min_x[0]][min_y[0]]} and a sigma of
            {sigmas[min_x[0]][min_y[0]]}.\n""")

    seed_mu = mus[min_x[0]][min_y[0]]
    seed_sigma = sigmas[min_x[0]][min_y[0]]

    # based on best mu and sigma get grouping of values around each
    # (the best_mu and best_si values can be found wihtout manually entering
    # them, I just need to write the code for it)

    best_mus = np.arange(seed_mu - 3, seed_mu + 3, 1)
    best_sis = np.arange(seed_sigma - 3, seed_sigma + 3, 1)

    best_mus, best_sis = np.meshgrid(best_mus, best_sis)
    iterable = zip(best_mus.flatten(), best_sis.flatten())

    # initialize dictionary and find best seed among grid
    grid_results = dict()
    grid_results['voxel'] = [x, y, z]
    grid_results['error'] = float('inf')
    for i, seed in enumerate(iterable):
        try:
            if seed[1] != 0:
                results = minimize(error_function,
                                   seed,
                                   (stim_space, norm_voxel, c_tr),
                                   method='Nelder-Mead')
            if results.fun < grid_results['error']:
                grid_results['error'] = results.fun
                grid_results['best_mu'] = results.x[0]
                grid_results['best_sigma'] = results.x[1]
        except ZeroDivisionError:
            print(f"Could not calculate for {seed}.")
            print("")

    # print results of grid search
    if print_status:
        print(f"Results for voxel: {grid_results['voxel']} ")
        print(f"Min error: {grid_results['error']}")
        print(f"Best mu: {grid_results['best_mu']}")
        print(f"Best sigma: {grid_results['best_sigma']}")

    done_voxel += 1

    # perform minimization on best seed
    best_seed = [grid_results['best_mu'], grid_results['best_sigma']]
    the_results = minimize(error_function,
                           best_seed,
                           (stim_space, norm_voxel, c_tr),
                           method="Nelder-Mead")
    if print_status:
        print(f"Best mu and sigma: {the_results.x}")
        print(f"Most minimized error: {the_results.fun}")

    all_results.append(grid_results)

    print(
        f"""-{round((done_voxel / total_voxel) * 100, 2)}% voxels complete-""")


toc = time.time()
print(f"Finished computing for all voxels. Time taken: {toc - tic} seconds.")

print("--------")
img_name = op.join(base_path, "testing_results.pickle")
with open(img_name, "wb") as f:
    pickle.dump(all_results, f)  # variable you want to save first then file
