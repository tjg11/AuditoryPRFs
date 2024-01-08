import numpy as np
from os import path as op
from dotenv import load_dotenv
from nilearn import plotting
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib as mpl
import pickle
import os
import json


def plot_results(subject_id,
                 z_slice,
                 focus_result="error",
                 check=True,
                 save_html=True,
                 show_plot=False):

    # load environment variables
    load_dotenv()

    # set path to GitHub directory
    path_main = os.getenv("MAIN_PATH")

    # set prf data path
    data_path = os.getenv("DATA_PATH")
    prf_path = op.join(
        data_path,
        "prf_results_1000_v"
    )

    # set path to json file storing mask file info for eac subject
    dict_path = os.path.join(
        path_main,
        "prfs",
        "roi_params.json"
    )

    # load json file to get mask parameters
    with open(dict_path, "r") as f:
        roi_params = json.load(f)

    # set roi loading parameters
    roi_area, roi_threshold = roi_params[subject_id]

    # set roi path
    roi_path = op.join(
        data_path,
        subject_id,
        "rois",
        f"{subject_id}_roi_size{roi_area}_p{roi_threshold}.pickle"
    )

    # load roi
    with open(roi_path, "rb") as f:
        roi_mask = pickle.load(f)

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
    out_name = op.join(prf_path, "results.nii.gz")

    # load results and print keys
    with open(img_name, "rb") as f:
        results = pickle.load(f)

    # convert to nifti and save
    print(out_name)
    img = nib.Nifti1Image(results[focus_result],
                          bold_img.affine,
                          bold_img.header)
    nib.save(img, out_name)

    if check:
        print("Keys in results file should be ['error', 'mus', 'sigmas'].")
        print(f"Actual results keys: {results.keys()}")

    # get error values
    if focus_result == "mus" or focus_result == "sigmas":
        result_values = results[focus_result]

        # apply roi mask to results
        print(result_values.shape, roi_mask.shape)
        print(type(result_values), type(roi_mask))
        result_values = result_values.astype(float)
        roi_mask = roi_mask.astype(bool)
        result_values[~roi_mask] = np.nan

        # threshold results based on stim space
        print(np.count_nonzero(~np.isnan(result_values)))
        result_values[result_values > 40] = np.nan
        result_values[result_values < -40] = np.nan
        print(np.count_nonzero(~np.isnan(result_values)))

        # check error values
        error = results["error"]
        print(f"Max stat is {np.nanmax(error)}.")
        print(f"Min stat is {np.nanmin(error)}.")
        # bool error values
        error[error <= 1] = 0
        error[error > 1] = 1
        error = error.astype(bool)

        # count number of non nan values
        print(np.count_nonzero(~np.isnan(result_values)))

        # threshold based on error
        result_values[error] = np.nan

        # count number of non nan values again
        print(np.count_nonzero(~np.isnan(result_values)))

        # convert array to nib image
        img = nib.Nifti1Image(
            result_values,
            bold_img.affine,
            bold_img.header)

        if check:
            print(f"Shape of {focus_result} is: {result_values.shape}")
            print(f"Max stat is {np.nanmax(result_values)}.")
            print(f"Min stat is {np.nanmin(result_values)}.")
            print(f"Median is {np.median(result_values)}")
        if save_html:
            html_path = os.path.join(
                data_path,
                "figures",
                f"{subject_id}_{focus_result}_overlay.html"
            )
            html = plotting.view_img(
                img,
                bg_img=bold_img.slicer[:, :, :, 100],
                threshold=0)
            html.save_as_html(html_path)

    # get values for histogram and remove zeros (for sigmas and mus)
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
        if show_plot:
            plt.show()

    else:
        result_values = results[focus_result]
        print(result_values.shape, type(result_values))
        roi_mask = roi_mask.astype(bool)
        result_values[~roi_mask] = np.nan
        if check:
            print(f"Shape of error is: {result_values.shape}")
            print(f"Max stat is {np.nanmax(result_values)}.")
            print(f"Min stat is {np.nanmin(result_values)}.")
            print(f"Median is {np.nanmedian(result_values)}")

        counts, bins = np.histogram(
            result_values)
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
            interpolation='nearest',
            alpha=0.75)
        ax2.set_title(f"{focus_result.capitalize()} for z-slice {z_slice}")
        plt.colorbar(im1, ax=ax2)
        # show plot
        plt.show()

    return


if __name__ == "__main__":
    # load environment variables
    load_dotenv()
    # load subject ids
    sub_ids = json.loads(os.getenv("SUB_IDS"))
    for sub in range(len(sub_ids)):
        # plot each result
        plot_results(sub_ids[sub], 26, focus_result="mus")
