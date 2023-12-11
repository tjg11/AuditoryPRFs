import numpy as np
from os import path as op
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib as mpl
import pickle
import os


def plot_results(subject_id,
                 z_slice,
                 focus_result="error",
                 check=True):

    # load environment variables
    load_dotenv()

    # set prf data path
    prf_path = os.getenv("DATA_PATH")
    prf_path = op.join(
        prf_path,
        subject_id,
        "prfs"
    )

    # set brain data path and load
    bf = ("_task-ampb_run-2_space-T1w_desc-preproc_bold.nii.gz")
    brain_path = os.getenv("ORIG_PATH")
    brain_path = op.join(
        brain_path,
        "derivatives",
        "fmriprep",
        subject_id,
        "ses-02",
        "func",
        f"{subject_id}_ses-02{bf}"
    )
    bold_img = nib.load(brain_path)
    bold_data = bold_img.get_fdata()
    print(f"Shape of brain data is {bold_data.shape}.")

    # pick time point
    time_point = 100

    # create slices
    slice1 = bold_data[:, :, z_slice, time_point]

    # get PRF results
    img_name = op.join(prf_path, f"prf_results_{subject_id}_final.pickle")

    # convert to nifti
    out_name = op.join(prf_path, "results.nii")

    # load results and print keys
    with open(img_name, "rb") as f:
        results = pickle.load(f)

    # convert to nifti and save
    print(out_name)
    img = nib.Nifti1Image(results[focus_result], np.eye(4))
    nib.save(img, out_name)

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
    plot_values[plot_values < -30] = np.nan
    plot_values[plot_values > 30] = np.nan
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
    plot_results("sub-NSxLxYKx1964", 25, focus_result="mus")
