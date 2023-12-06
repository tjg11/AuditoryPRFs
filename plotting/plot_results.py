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
    data_path = os.getenv("DATA_PATH")
    prf_path = op.join(
        data_path,
        subject_id,
        "prfs"
    )

    roi_path = op.join(
        data_path,
        subject_id,
        "rois",
        "sub-NSxLxYKx1964_tarea50_p0.0001_roi.pickle"
    )

    with open(roi_path, "rb") as f:
        roi_mask = pickle.load(f)

    

    # set brain data path and load
    bf = ("_task-ampb_run-2_space-T1w_desc-preproc_bold.nii.gz")
    brain_path = os.getenv("ORIG_PATH")
    brain_path = op.join(
        brain_path,
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

    # apply roi mask to results
    print(result_values.shape, roi_mask.shape)
    print(type(result_values), type(roi_mask))
    result_values = result_values.astype(float)
    roi_mask = roi_mask.astype(bool)
    result_values[~roi_mask] = np.nan

    # threshold results based on stim space
    result_values[result_values > 40] = np.nan
    result_values[result_values < -40] = np.nan

    # threshold based on error
    result_values[results['error'] < 0.5] = np.nan

    if check:
        print(f"Shape of error is: {result_values.shape}")
        print(f"Max stat is {np.nanmax(result_values)}.")
        print(f"Min stat is {np.nanmin(result_values)}.")
        print(f"Median is {np.median(result_values)}")

    # get values for histogram and remove zeros
    counts, bins = np.histogram(result_values, bins=np.arange(-30, 30, 1))
    # counts = counts[1:]
    # bins = bins[1:]

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
    # plot_values[plot_values < 0.001] = np.nan

    im1 = ax2.imshow(
        plot_values,
        cmap=cmap,
        vmin=-30,
        vmax=30,
        interpolation='nearest',
        alpha=0.75)
    ax2.set_title(f"{focus_result.capitalize()} for z-slice {z_slice}")
    plt.colorbar(im1, ax=ax2)
    # show plot
    plt.show()
    return


if __name__ == "__main__":
    plot_results("sub-NSxLxYKx1964", 26, focus_result="mus")
