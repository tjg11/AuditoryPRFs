import os
import json
import pickle
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from dotenv import load_dotenv


def hemi_split(
        img_array: list,
        l_hemi: list,
        r_hemi: list,
        hist_bins=np.arange(-30, 30, 1),
        show_plots=False
):
    """
    Takes an array of images and two segmentation masks, one for the left
    hemisphere and one for the right hemisphere. Plots a histogram of the
    values contained in each hemisphere for all subjects, and returns the mean
    values in the left and right hemispheres, respectivley.
    """
    # copy img array for getting both left and right hemisphere counts
    l_hemi_arr = img_array.copy()
    r_hemi_arr = img_array.copy()

    # recast mask images
    l_hemi_arr = l_hemi_arr.astype(float)
    r_hemi_arr = r_hemi_arr.astype(float)
    l_hemi = l_hemi.astype(bool)
    r_hemi = r_hemi.astype(bool)

    # mask images in l_hemi array
    l_hemi_arr[~l_hemi] = np.nan

    # mask images in r_hemi array
    r_hemi_arr[~r_hemi] = np.nan

    # get historgram for l_counts
    l_counts, bins = np.histogram(l_hemi_arr, bins=hist_bins)

    # get histogram for r_counts
    r_counts, bins = np.histogram(r_hemi_arr, bins=hist_bins)

    # get mean for l_counts
    l_mean = np.round(np.mean(l_counts), 2)

    # get mean deviation for r_counts
    r_mean = np.round(np.mean(r_counts), 2)

    # plot both historgrams
    if show_plots:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        ax1.stairs(l_counts, bins, fill=True)
        ax1.set_title(f"Left hemi: mean = {l_mean}")

        ax2.stairs(r_counts, bins, fill=True)
        ax2.set_title(f"Right hemi: mean = {r_mean}")

        plt.show()

    # return left and right hemisphere means
    return l_mean, r_mean


if __name__ == '__main__':

    # load environment variables
    load_dotenv()

    # set path variables
    paths_ampb = os.getenv("ORIG_PATH")
    paths_data = os.getenv("DATA_PATH")
    paths_main = os.getenv("MAIN_PATH")

    # load file names
    seg_name = os.getenv("SEG_NAME")

    # load subject ROI mask data
    paths_roi = os.path.join(
        paths_main,
        "prfs",
        "roi_params.json"
    )
    with open(paths_roi, "r") as f:
        roi_data = json.load(f)

    # load subject ids
    sub_ids = json.loads(os.getenv("SUB_IDS"))

    # iterate through subjects to test and store means
    l_means = []
    r_means = []
    for sub_id in sub_ids:

        # set path for subject ROI mask
        roi_size = roi_data[sub_id][0]
        roi_thrs = roi_data[sub_id][1]
        paths_subroi = os.path.join(
            paths_data,
            sub_id,
            "rois",
            f"{sub_id}_roi_size{roi_size}_p{roi_thrs}.pickle"
        )

        # load subject ROI mask
        with open(paths_subroi, "rb") as f:
            roi_mask = pickle.load(f)

        # load mu results for subject (and error values)
        path_results = os.path.join(
            paths_data,
            f"prf_results_{sub_id}_final.pickle"
        )
        with open(path_results, "rb") as f:
            results = pickle.load(f)
        error_values = results['error']
        results = results['mus']

        # clean up results before passing
        results[results > 40] = np.nan
        results[results < -40] = np.nan
        results[error_values > 1] = np.nan

        # set paths for segmentation and label files
        paths_seg = os.path.join(
            paths_ampb,
            "derivatives",
            "fmriprep",
            sub_id,
            "ses-01",
            "func",
            f"{sub_id}_ses-01_task-ptlocal_run-1_space-T1w_desc-aseg_dseg.nii.gz"
        )
        paths_label = os.path.join(
            paths_ampb,
            "derivatives",
            "fmriprep",
            "desc-aseg_dseg.tsv"
        )

        # load segmentation and create two copies for two masks
        segmentation = nib.load(paths_seg).get_fdata()
        l_seg = segmentation.copy()
        r_seg = segmentation.copy()

        # load label file
        label_csv = pd.read_csv(paths_label, sep='\t')

        # set regions of interest
        regions = ['Left-Cerebral-Cortex', 'Right-Cerebral-Cortex']

        # find target indicies in label csv
        target_idxs = []
        for region in regions:
            target = label_csv[label_csv.name == region]
            target_idx = target.iloc[0][0]
            target_idxs.append(target_idx)

        # set left and right hemisphere indicies
        l_idx = target_idxs[0]
        r_idx = target_idxs[1]

        # use left index to create left hemisphere mask
        l_seg[l_seg == l_idx] = -1
        l_seg[l_seg != -1] = 0
        l_seg[l_seg == -1] = 1

        # use right index to create right hemisphere mask
        r_seg[r_seg == r_idx] = -1
        r_seg[r_seg != -1] = 0
        r_seg[r_seg == -1] = 1

        # filter left hemisphere mask using roi mask
        l_seg = l_seg * roi_mask

        # filter right hemisphere mask using roi mask
        r_seg = r_seg * roi_mask

        # plot the histogram!
        l_count, r_count = hemi_split(
            results,
            l_seg,
            r_seg
        )

        # store the means
        l_means.append(l_count)
        r_means.append(r_count)
    # perform ANOVA on the means of each group
    print(f_oneway(l_means, r_means))
