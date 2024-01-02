import os
import json
import pickle
import pandas as pd
import numpy as np
import nibabel as nib
from get_rois_reworked import get_rois
from dotenv import load_dotenv
import time
from settings import P_THRESHOLDS, ROI_SIZES


def batch_rois(
    sub_id,
    seg_path,
    mask_path,
    label_path,
    save_path,
    labels,
    p_value_path,
    z_score_path,
    p_thresholds,
    roi_sizes,
):
    """
    Takes all subject related parmaters, such as paths to various image files
    and lists containing parameters for defining ROI sizes, and uses get_rois
    to generate multiple ROI masks. In order for this function to work, all of
    paths entered need to exist, and the image files they point to need to have
    the same sizes as one another. This function is not necessary to use
    get_rois, however it can be more efficient if the paths are set correctly.
    """

    # load segmentation image
    seg_img = nib.load(seg_path).get_fdata().astype(int)

    # load mask image
    mask_img = nib.load(mask_path).get_fdata()

    # load label csv
    label_csv = pd.read_csv(label_path, sep='\t')

    # load p-values
    p_values = nib.load(p_value_path).get_fdata()

    # load z-scores
    z_scores = nib.load(z_score_path).get_fdata()

    # print stats for p_values and z_scores
    p_mean = np.mean(p_values)
    z_mean = np.mean(z_scores)

    p_max = np.amax(p_values)
    z_max = np.amax(z_scores)

    p_min = np.amin(p_values)
    z_min = np.amin(z_scores)

    print(f"P-VALUES: min = {p_min} max = {p_max} mean = {p_mean}")
    print(f"Z-SCORES: min = {z_min} max = {z_max} mean = {z_mean}")

    # get an array of all combinations of p-values and roi sizes
    complete_thresholds = np.vstack(np.meshgrid(roi_sizes, p_thresholds))
    complete_thresholds = complete_thresholds.reshape(2, -1).T

    # create dictionary to store label sizes
    count_dict = {}

    # create mask for each parameter combination
    for roi_size, p_threshold in complete_thresholds:
        roi_img, roi_count = get_rois(
            seg_img,
            mask_img,
            label_csv,
            labels,
            p_values,
            z_scores,
            p_threshold,
            roi_size
        )
        dict_key = str(roi_size) + " " + str(p_threshold)
        count_dict[dict_key] = roi_count
        # save the mask
        file_name = f"{sub_id}_roi_size{roi_size}_p{p_threshold}.pickle"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        new_path = os.path.join(save_path, file_name)
        with open(new_path, "wb") as f:
            pickle.dump(roi_img, f)

    # return subject id and count dictionary
    return sub_id, count_dict


if __name__ == "__main__":
    # load environment variables
    load_dotenv()

    # set experiment data path
    ampb_path = os.getenv("ORIG_PATH")

    # set analysis data path
    data_path = os.getenv("DATA_PATH")

    # get list of subject IDS
    sub_ids = json.loads(os.getenv("SUB_IDS"))

    # get list of roi areas to include
    roi_areas = json.loads(os.getenv("ROI_AREAS"))

    # get file names for segmentation and brain mask files
    seg_name = os.getenv("SEG_NAME")
    mask_name = os.getenv("MASK_NAME")
    p_name = os.getenv("P_NAME")
    z_name = os.getenv("Z_NAME")

    # initialize dataframe to store all ROI sizes
    roi_counts = pd.DataFrame()

    # iterate through subject ids and create
    for sub_id in sub_ids:
        # get segmentation path
        seg_path = os.path.join(
            ampb_path,
            "derivatives",
            "fmriprep",
            sub_id,
            "ses-01",
            "func",
            f"{sub_id}{seg_name}"
        )

        # get mask path
        mask_path = os.path.join(
            ampb_path,
            "derivatives",
            "fmriprep",
            sub_id,
            "ses-01",
            "func",
            f"{sub_id}{mask_name}"
        )

        # get label path
        label_path = os.path.join(
            ampb_path,
            "derivatives",
            "fmriprep",
            "desc-aseg_dseg.tsv"
        )

        # get p-value path
        p_value_path = os.path.join(
            data_path,
            sub_id,
            "contrast_maps",
            f"{sub_id}{p_name}"
        )

        # get z-score path
        z_score_path = os.path.join(
            data_path,
            sub_id,
            "contrast_maps",
            f"{sub_id}{z_name}"
        )

        # set save path
        save_path = os.path.join(
            data_path,
            sub_id,
            "rois"
        )

        sub_id, count_dict = batch_rois(
            sub_id,
            seg_path,
            mask_path,
            label_path,
            save_path,
            roi_areas,
            p_value_path,
            z_score_path,
            P_THRESHOLDS,
            ROI_SIZES
        )

        # display results in a readable way
        print(f"START OF {sub_id}:")
        indicies = []
        for key in count_dict.keys():
            print(f"PARAMS: {key} SIZE - {count_dict[key]}")
            indicies.append(str(key))
        df = pd.DataFrame(data=count_dict, index=[sub_id])
        roi_counts = pd.concat([roi_counts, df], axis=0)
        print(roi_counts.head())
    # set path to save csv and save
    count_path = os.path.join(
        data_path,
        "roi_counts.csv"
    )
    roi_counts.to_csv(count_path)
