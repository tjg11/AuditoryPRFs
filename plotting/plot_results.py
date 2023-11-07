import numpy as np
from os import path as op
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib as mpl
import pickle
import os
import time


def plot_results(subject_id,
                 z_slice,
                 focus_result="error",
                 check=True):

    # set base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    # set brain data path and load
    brain_path = op.join(
        base_path,
        "..",
        "..",
        "AMPB",
        "data",
        subject_id,
        "ses-02",
        "func",
        (subject_id + "_ses-02_task-ampb_run-1_bold.nii.gz"))
    bold_img = nib.load(brain_path)
    time.sleep(10)
    bold_data = bold_img.get_fdata()
    print(f"Shape of brain data is {bold_data.shape}.")

    # pick time point
    time_point = 100

    # create slices
    slice1 = bold_data[:, :, z_slice, time_point]

    # get PRF results
    img_name = op.join(base_path, f"prf_results_{subject_id}_final.pickle")

    # load results and print keys
    with open(img_name, "rb") as f:
        results = pickle.load(f)

    if check:
        print("Keys in results file should be ['error', 'mus', 'sigmas'].")
        print(f"Actual results keys: {results.keys()}")

    # get error values
    result_values = results[focus_result]
    if check:
        print(f"Shape of error is: {result_values.shape}")
        print(f"Max stat is {np.amax(result_values)}.")
        print(f"Min stat is {np.amin(result_values)}.")
        print(f"Median is {np.median(result_values)}")

    # get values for histogram and remove zeros
    counts, bins = np.histogram(result_values)
    counts = counts[1:]
    bins = bins[1:]

    # plot histogram and map
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # histogram
    ax1.stairs(counts, bins, fill=True)
    ax1.set_title(f"Distribution of {focus_result}")

    # map
    cmap = mpl.cm.get_cmap('jet').copy()
    # cmap.set_under(color='grey')

    ax2.imshow(slice1, cmap='Greys')

    plot_values = result_values[:, :, z_slice]
    # plot_values[plot_values < -30] = np.nan
    # plot_values[plot_values > 30] = np.nan
    # plot_values[np.abs(plot_values) < 0.001] = np.nan
    plot_values[plot_values < 0.001] = np.nan

    im1 = ax2.imshow(
        plot_values,
        cmap=cmap,
        # vmin=0.0000000000001,
        vmin=0.001,
        interpolation='nearest',
        alpha=0.75)
    ax2.set_title(f"{focus_result.capitalize()} for z-slice {z_slice}")
    plt.colorbar(im1, ax=ax2)
    # show plot
    plt.show()
    return


if __name__ == "__main__":
    plot_results("sub-NSxGxHKx1965", 25, focus_result="error")
