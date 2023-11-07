# Author: Taylor Garrison
# Description: Designed to utilize outputs from stimulus_creation.py and
# convolve_hrf.py to determine and visualize population receptive fields (PRFs)
# based on parametized predicted time course.
# Version 2.0 of find_prf with better time complexity (ideally O(n^2))

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


# -------------------------------------------------------------------
# Print sanity check statements and plots?
sanity_check = True
# sanity_check = False
plot_timecourses = True
# plot_timecourse = False
print_status = False
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Pick a subject
subject_id = "sub-EBxGxCCx1986"
# -------------------------------------------------------------------

# -------------------------------------------------------------------
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
        f"""Total length of run should be 416 seconds and total rows per second
        should be 100. Shape should be 41600, 9.")
        Shape is: {loaded_record.shape}.
        Total length of run: {loaded_record.shape[0] / 100} seconds.
        Total rows per second: {loaded_record.shape[0] / 416} rows.""")

# -------------------------------------------------------------------

# -------------------------------------------------------------------


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


# Minimized error function: take predicted time courses and actual time course,
# find minimum error and return
def find_min(convolved_stim,
             stim_space,
             params,
             voxel_t,
             mus,
             sigmas,
             error_matrix):
    for i, params in enumerate(params):

        mu, sigma = params  # current parameters

        func = generate_gaussian(stim_space, mu, sigma)
        pred_t = np.matmul(convolved_stim, func)
        corr = pearsonr(voxel_t, pred_t).statistic
        error_matrix[i] = -corr

        error_matrix = error_matrix.reshape(mus.shape)

        min_x, min_y = np.where(error_matrix == np.nanmin(error_matrix))

        seed_mu = mus[min_x[0]][min_y[0]]
        seed_sigma = sigmas[min_x[0]][min_y[0]]

        return seed_mu, seed_sigma


# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Fuction starts here:
# Parameters: sub_id, session number, in_path, out_path (default to same
# directory), x_padding(tuple, default to (0, 0), y_padding(tuple, default to
# (0,0), x_start/end (tuple, optional, default to False), same for y and z,
# Still need to add functionality for changing path
# ---------------------
def find_prf(subject_id,
             ses_number,
             mask=None,
             x_padding=(0, 0),
             y_padding=(0, 0)):

    base_path = os.path.dirname(os.path.abspath(__file__))
    img_name = op.join(base_path,
                       subject_id,
                       f"convolved_{subject_id}_{ses_number}.pickle")
    with open(img_name, "rb") as f:
        loaded_record = pickle.load(f)

    # Sanity check the loaded file
    if sanity_check:
        print(
            f"""Total length of run should be 416 seconds and total rows per
            second should be 100. Shape should be 41600, 9.
            Shape is: {loaded_record.shape}.
            Total length of run: {loaded_record.shape[0] / 100} seconds.
            Total rows per second: {loaded_record.shape[0] / 416} rows.""")

    # Pad beginning of stimulus with 8 seconds of zeros (so 800 rows).
    # Padding parameters = (before, after)
    x_pad = (x_padding[0], x_padding[1])
    y_pad = (y_padding[0], y_padding[1])

    padded_stim = np.pad(loaded_record, (x_pad, y_pad), mode="constant")

    # Sanity check the padded file
    if sanity_check:
        print(
            f"""Total length of run should be 430 seconds and total rows per
            second should be 100. Shape should be 43000, 9.
            Shape is: {padded_stim.shape}.
            Total length of run: {padded_stim.shape[0] / 100} seconds.
            Total rows per second: {padded_stim.shape[0] / 430} rows.""")

    # Interpolate to TR sampling rate (2 seconds).

    total_sec = padded_stim.shape[0] / 100  # total time
    dt = 0.01  # sampling rate
    ts = np.arange(0, total_sec, dt)  # time vector

    tr_func = scipy.interpolate.interp1d(ts, padded_stim, axis=0)
    tr = np.arange(0, ts[-1], 2)
    c_tr = tr_func(tr)

    if sanity_check:
        print("Shape should be 215,9. There should be 215 trials.")
        print(f"Shape is {c_tr.shape}.")
        print(f"There are {c_tr.shape[0]} trials.")

    # Load brain data #TODO: CHANGE TO DERIVATIVES
    brain_file = (f"_task-ampb_run-{ses_number[1]}_bold.nii.gz")
    brain_path = op.join(base_path,
                         "..",
                         "..",
                         "AMPB",
                         "data",
                         subject_id,
                         "ses-02",
                         "func",
                         f"{subject_id}_ses-02{brain_file}")
    bold_img = nib.load(brain_path)
    bold_data = bold_img.get_fdata()
    print(f"Shape of brain data is {bold_data.shape}.")

    # This part should now mask the data given a mask
    if mask is not None:
        with open(mask, "rb") as f:
            brain_mask = pickle.load(f)
        for slice in range(bold_data.shape[3]):
            bold_data[:, :, :, slice] = bold_data[:, :, :, slice] * brain_mask
    print(f"SHAPE IS {bold_data.shape} ")

    # Generate list of coords to analyze
    x, y, z = np.nonzero(brain_mask)
    coords = []
    for idx in range(len(x)):
        coords.append((x[idx], y[idx], z[idx]))

    error_results = np.zeros((brain_mask.shape))
    mu_results = np.zeros((brain_mask.shape))
    sigma_results = np.zeros((brain_mask.shape))

    # error_results = mu_results = sigma_results = np.zeros((x, y, z))
    # shallow vs. deep copy of the variable, check

    # Generate parameter grid
    mus = np.arange(-31, 30, 6)
    sigmas = np.arange(-31, 30, 6)
    mus, sigmas = np.meshgrid(mus, sigmas)

    error_matrix = np.zeros(mus.shape).flatten()
    all_params = zip(mus.flatten(), sigmas.flatten())
    stim_space = np.linspace(-30, 30, 9)

    voxels_done = 0
    voxels_total = len(coords)
    print(f"There are {voxels_total} voxels to be analyzed.")
    print("Check this against the number of voxels in the mask.")

    tic = time.time()

    for coord in coords:

        # Set timecourse
        x, y, z = coord
        voxel = bold_data[x, y, z, :]
        min_voxel = (voxel - np.amin(voxel))
        max_voxel = (np.amax(voxel) - np.amin(voxel))
        norm_voxel = min_voxel / max_voxel

        print(f"\n-----Starting for {x, y, z} voxel------\n")

        # Determine best seed
        try:
            seed_mu, seed_sigma = find_min(
                c_tr,
                stim_space,
                all_params,
                voxel,
                mus,
                sigmas,
                error_matrix)
        except TypeError:
            seed_mu, seed_sigma = 0, 0

        # Create new grid based on best seed
        best_mus = np.arange(seed_mu - 3, seed_mu + 3, 1)
        best_sis = np.arange(seed_sigma - 3, seed_sigma + 3, 1)

        best_mus, best_sis = np.meshgrid(best_mus, best_sis)
        iterable = zip(best_mus.flatten(), best_sis.flatten())

        # Find best seed out of grid of best seeds
        temp_error = float('inf')
        temp_mu = 0
        temp_sig = 0
        for i, seed in enumerate(iterable):
            try:
                if seed[1] != 0:
                    results = minimize(error_function,
                                       seed,
                                       (stim_space, norm_voxel, c_tr),
                                       method='Nelder-Mead')
                if results.fun < temp_error:
                    temp_error = results.fun
                    temp_mu = results.x[0]
                    temp_sig = results.x[1]
            except ZeroDivisionError:
                print(f"Could not calculate for {seed}.")
                print("")
                # might want to include NaN as the error for uncalculable
                # something that is not a numeric type

        # Perform minimization on best seed
        # try to use named arguments when possible for clarity
        best_seed = [temp_mu, temp_sig]
        the_results = minimize(
            error_function,
            best_seed,
            (stim_space, norm_voxel, c_tr),
            method="Nelder-Mead"
        )  # not in for loop

        # Save results of final minimization
        error_results[x, y, z] = the_results.fun
        mu_results[x, y, z] = the_results.x[0]
        sigma_results[x, y, z] = the_results.x[1]

        voxels_done += 1
        complete_voxel = round((voxels_done / voxels_total) * 100, 2)
        print(f"---------{complete_voxel}% voxels complete----------")

    # Save results out as one dictionary
    prf_results = dict()
    prf_results["error"] = error_results
    prf_results["mus"] = mu_results
    prf_results["sigmas"] = sigma_results

    img_name = op.join(base_path, f"prf_results_{subject_id}_final.pickle")
    with open(img_name, "wb") as f:
        pickle.dump(prf_results, f)

    toc = time.time()

    print(f"Time taken: {toc - tic} seconds.")
    return


if __name__ == '__main__':
    print(find_prf("sub-NSxGxHKx1965",
                   "02",
                   op.join("rois",
                           "sub-NSxGxHKx1965",
                           "sub-NSxGxHKx1965_tarea25.0_p0.05_roi.pickle"),
                   x_padding=(800, 600)))
