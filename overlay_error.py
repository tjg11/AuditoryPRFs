# testing
import numpy as np
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

scaled_zero = False
# Pick a subject
subject_id = "sub-EBxGxCCx1986"

base_path = os.path.dirname(os.path.abspath(__file__))

brain_path = op.join(base_path, "..", "..", "AMPB", "data",subject_id, "ses-02", "func", (subject_id + "_ses-02_task-ampb_run-1_bold.nii.gz"))
bold_img = nib.load(brain_path)
bold_data = bold_img.get_fdata()
print(f"Shape of brain data is {bold_data.shape}.")

time_point = 100
z_slice1 = 20
z_slice2 = 25
z_slice3 = 27
z_slice4 = 30

slice1 = bold_data[ :, :, z_slice1, time_point]
slice2 = bold_data[ :, :, z_slice2, time_point]
slice3 = bold_data[ :, :, z_slice3, time_point]
slice4 = bold_data[ :, :, z_slice4, time_point]

base_path = os.path.dirname(os.path.abspath(__file__))

img_name = op.join(base_path, f"prf_results_{subject_id}_updated_function.pickle")
# img_name = op.join(base_path, f"prf_results_{subject_id}_final.pickle")

# x_start = 1
# x_end = 71

# y_start = 1
# y_end = 71

# z_start = 21
# z_end = 23

# z_slices = z_end - z_start
# print(z_slices)

# x_ax = x_end - x_start
# y_ax = y_end - y_start

# error_values = np.zeros((72, 72))
# print(error_values.shape)
# print(x_ax, y_ax, x_ax * y_ax)
with open(img_name, "rb") as f:
    results = pickle.load(f)

# for voxel in results:
#     print(voxel["error"])

# print(len(results) / 2)
# for voxel in results:
#     if scaled_zero:
#         # plot scaled to zero
#         x_coord = voxel["voxel"][0] - x_start
#         y_coord = voxel["voxel"][1] - y_start
#     else:
#         # plot same scale as image
#         x_coord = voxel["voxel"][0]
#         y_coord = voxel["voxel"][1]

#     error_values[x_coord][y_coord] = voxel["error"]
print(results.keys())
error_values = results["mus"]
print(error_values)

# print(error_values[:, :, z_slice])


# fig, ax = plt.subplots(2, 2)

# ax[0, 0].imshow(slice1, cmap='Greys')
# ax[0, 0].imshow(error_values[:, :, z_slice1], cmap='jet', interpolation='nearest', alpha=0.5, vmin=0.08)

# ax[0, 1].imshow(slice2, cmap='Greys')
# ax[0, 1].imshow(error_values[:, :, z_slice2], cmap='jet', interpolation='nearest', alpha=0.5, vmin=0.08)

# ax[1][0].imshow(slice3, cmap='Greys')
# ax[1][0].imshow(error_values[:, :, z_slice3], cmap='jet', interpolation='nearest', alpha=0.5, vmin=0.08)

# ax[1][1].imshow(slice4, cmap='Greys')
# ax[1][1].imshow(error_values[:, :, z_slice4], cmap='jet', interpolation='nearest', alpha=0.5, vmin=0.08)


# plt.show()

plt.imshow(slice1, cmap='Greys')
plt.imshow(error_values[:, :, z_slice1], cmap='jet', interpolation='nearest', vmin=-40, vmax=40, alpha=0.75)
plt.colorbar()
plt.show()

# mus = results["mus"]
# plt.imshow(slice, cmap='Greys')
# plt.imshow(mus[:, :, z_slice], cmap='jet', interpolation='nearest', alpha=0.5, vmin=0.08, vmax=20)
# plt.colorbar()
# plt.show()


# testing meshgrid for use with creating coordinate matrix

# x = np.arange(40, 61)
# y = np.arange(10, 31)
# z = np.arange(10, 31)

# print(x)
# print(y)
# print(z)

# xv, yv, zv = np.meshgrid(x, y, z,  indexing='ij')


# gridvalues_combined_tidy = np.vstack([xv.flatten(), yv.flatten(), zv.flatten()]).T

# count = 0
# for coord in gridvalues_combined_tidy:
#     count += 1
#     print(coord)

# print(count)

# Plotting
# x = np.array([x["error"] for x in results])
# plt.hist(x)
# plt.show()