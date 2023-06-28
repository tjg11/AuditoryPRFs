### Author: Taylor Garrison
### Description: Designed to utilize outputs from stimulus_creation.py and convolve_hrf.py to
### determine and visualize population receptive fields (PRFs) based on parametized predicted
### time course.

from os import path as op
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
import scipy
import glob
import scipy.stats

def generate_gaussian(x, b, c):
    return np.exp(-(((x - b) ** 2) / (2 * c ** 2)))

# Load one data file and check shape for sanity check
img_name = op.join("sub-EBxGxCCx1986", "convolved_sub-EBxGxCCx1986_1.pickle")
with open(img_name, "rb") as f:
    loaded_record = pickle.load(f)
print(loaded_record.shape)
print(f"Total length of run: {loaded_record.shape[0] / 100} seconds.")
print(f"Total rows per second: {loaded_record.shape[0] / 416} rows.")

# Pad beginning of stimulus with 8 seconds of zeros (so 800 rows).

# Padding parameters = (before, after)
x_pad = (800, 600)
y_pad = (0, 0)

padded_stim = np.pad(loaded_record, (x_pad, y_pad), mode='constant')
total_sec = padded_stim.shape[0] / 100

print(padded_stim.shape)
print(f"Total length of run: {total_sec} seconds.")
print(f"Total rows per second: {padded_stim.shape[0] / total_sec} rows.")

# Interpolate to TR sampling rate (2 seconds).

dt = 0.01 # sampling rate
ts = np.arange(0, total_sec, dt) # time vector

tr_func = scipy.interpolate.interp1d(ts, padded_stim, axis = 0)
tr = np.arange(0, ts[-1], 2)
c_tr = tr_func(tr)

print(c_tr.shape)
print(f"There are {c_tr.shape[0]} each 2 seconds long for {c_tr.shape[0] * 2} seconds total.")
plt.plot(c_tr)

# Load brain data

brain_path = op.join("..", "..", "AMPB", "data","sub-EBxGxCCx1986", "ses-02", "func", "sub-EBxGxCCx1986_ses-02_task-ampb_run-1_bold.nii.gz")
bold_img = nib.load(brain_path)
bold_data = bold_img.get_fdata()
print(bold_data.shape)

# Pick a voxel
x_coord = 50
y_coord = 22
z_coord = 21
fake_brain = bold_data[x_coord, y_coord, z_coord, :]
plt.plot(fake_brain)

mu = 0
sigma = 3
x = np.linspace(-30, 30, 9)

# Generate multiple gaussians and find minimum correlation
import numpy as np

x = np.linspace(-30, 30, 9)

mus    = np.linspace(-30, 30, 7)
sigmas = [0.1, 1, 5, 10, 13, 15]
mus, sigmas = np.meshgrid(mus, sigmas)

error_matrix = np.zeros(mus.shape).flatten()
iterable = zip(mus.flatten(), sigmas.flatten())
for i, params in enumerate(iterable):
    mu, sigma = params # current parameters

    func = generate_gaussian(x, mu, sigma)
    pred_t = np.matmul(c_tr, func)
    corr = pearsonr(fake_brain, pred_t).statistic
    error_matrix[i] = -corr
error_matrix = error_matrix.reshape(mus.shape)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

ax.plot_wireframe(mus, sigmas, error_matrix)
ax.set_xlabel('MU')
ax.set_ylabel('SIGMA')
ax.set_zlabel('ERROR')
plt.show()