import os
import pickle
import time
import json
from os import path as op
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation
import warnings
from dotenv import load_dotenv


# Step 1: get data / files
# Step 2: load data / files
# Step 2b: things needed - brain mask, segmentation?, zmap, and pvalues
# Step 3: resample / binarize images that need to be resamples (brain mask)
# Step 4: get region labels
# Step 5: apply filters to labels and save coordinates


def get_rois(sub_id,
             target_area=50,
             p_threshold=.0001,
             seg_areas=['Left-Cerebral-Cortex',
                        'Right-Cerebral-Cortex'],
             paths_base=None,
             paths_main=None,
             paths_sub=None,
             save_file=True,
             ):
    """
    Creates an ROI map for a specific participant based off of the following
    parameters: target area (# smallest number of voxels required to be
    included), p_threshold (largest p value in order for a voxel to be
    included), seg_areas (areas of the brain to included based off of labels
    from segmentation file generated by FMRIPrep). Saves ROI out as pickle
    file, where 0s represent voxels not included in the ROI and 1s represent
    voxels included in the ROI. Returns the number of non-zero voxels in the
    ROI.
    """

    warnings.filterwarnings("ignore")

    # load environment variables
    load_dotenv()

    # set paths
    if paths_base is None:
        paths_base = os.getenv("BASE_PATH")
    if paths_main is None:
        paths_main = os.getenv("DATA_PATH")
    if paths_sub is None:
        paths_sub = os.getenv("ORIG_PATH")
        paths_sub = op.join(
            paths_sub,
            "fmriprep",
            sub_id,
            "ses-01",
            "func"
        )

    # set data paths
    paths_data_zmaps = op.join(paths_main, sub_id, "contrast_zmaps")
    paths_data_pvalue = op.join(paths_main, sub_id, "contrast_niftis")

    # load z-maps with original data and check for file not found
    f_name = op.join(paths_data_zmaps,
                     f"{sub_id}_sound-silent.nii") #TODO: user defined contrast?
    if op.exists(f_name):
        print("Z-map file exists!")
    else:
        print("Z-map file doesn't exist! Check the path for errors.")
        return
    z_maps = nib.load(f_name).get_fdata()

    # load brain mask and resample and check for file not found
    file_text = "ses-01_task-ptlocal_run-1_space-T1w_desc-brain_mask.nii.gz"
    f_name = op.join(paths_sub,
                     f"{sub_id}_{file_text}")
    if op.exists(f_name):
        print("Brain mask file exists!")
    else:
        print("Brain mask file doesn't exist! Check the path for errors.")
        print(f_name)
        return
    rb_mask = nib.load(f_name)  # original mask
    bin_rb_mask = rb_mask != 0  # binarized resampled b_mask

    # load segmentation
    ft = f"{sub_id}_ses-01_task-ptlocal_run-1_space-T1w_desc-aseg_dseg.nii.gz"
    f_name = op.join(paths_sub,
                     ft)
    if op.exists(f_name):
        print("Segmentation file exists!")
    else:
        print("Segmentation file does not exist. Check for path errors.")
    rseg = nib.load(f_name)
    rseg = rseg.get_fdata().astype(int)

    # load segmentation legend and find target indicies based on seg_areas
    f_name = op.join(os.getenv("ORIG_PATH"),
                     "fmriprep",
                     "desc-aseg_dseg.tsv")
    legend = pd.read_csv(f_name, sep='\t')
    # rename column b/c index is a keyword
    legend = legend.rename(
        columns={
            'index': 'legend_index'
        })

    target_idxs = []
    # loop through and find target indicies
    for seg_area in seg_areas:
        target = legend[legend.name == seg_area]
        target_idx = target.iloc[0][0]
        target_idxs.append(target_idx)

    # create segmentation mask based on target_idxs
    # set wanted areas to 0 and then flip
    for target_idx in target_idxs:
        fseg = rseg.copy()
        fseg[fseg == target_idx] = 0
        rseg = rseg * fseg
    rseg[rseg != 0] = 1
    rseg = 1 - rseg  # flip

    # load p-values and check for file not found
    f_name = op.join(paths_data_pvalue,
                     f"{sub_id}_sound-silent.nii") #TODO: can save as .nii.gz to save storage space
    if op.exists(f_name):
        print("P-value file exists!")
    else:
        print("P-value file does not exist! Check the path for errors.")
        return
    p_values = nib.load(f_name).get_fdata()

    # threshold z-map using p-values and binarize
    z_maps[p_values > p_threshold] = 0
    bin_zmaps = z_maps != 0

    # create  and filter labels based on target area
    label_image = label(bin_zmaps, connectivity=1)
    for region in regionprops(label_image):
        if region.area < target_area:
            label_image[label_image == region.label] = 0

    # binarize label image, dilate, and apply brain + seg mask
    bin_label_image = label_image != 0
    dilated_label_image = binary_dilation(bin_label_image)
    brain_mask_label_image = dilated_label_image * bin_rb_mask
    final_label_image = brain_mask_label_image * rseg
    vox_count = np.count_nonzero(final_label_image)
    print(f"Subject [{sub_id}] complete, see voxel count below.")
    print(f"The number of voxels in the final label image: {vox_count} ")

    # save files
    if save_file:
        file_text = f"{sub_id}_tarea{target_area}_p{p_threshold}_roi.pickle"
        f_path = op.join(paths_main,
                         sub_id,
                         "rois")
        f_name = op.join(f_path, file_text)
    if not op.exists(f_path):
        os.mkdir(f_path)
    with open(f_name, "wb") as f:
        pickle.dump(final_label_image, f)
    print(f"Finished [{sub_id}] successfully\n\n")
    # return the number of voxels in the ROI
    return vox_count


def batch_rois(
        sub_ids: list,
        t_areas: list,
        p_threshs: list,
        seg_areas=['Left-Cerebral-Cortex',
                   'Right-Cerebral-Cortex'],
        paths_base=None,
        paths_main=None,
        paths_sub=None,
        save_file=True,):
    """Performs multiple ROI calculations given a list of subject IDS, area
    parameters, threshold parameters, and segmentation areas. Creates each
    combination of """
    # create grid of areas and thresholds
    complete_thresholds = np.vstack(np.meshgrid(t_areas, p_threshs))
    complete_thresholds = complete_thresholds.reshape(2, -1).T

    # track number of faliures
    fail_count = 0

    for sub_id in sub_ids:
        for t_area, p_thresh in complete_thresholds:
            try:
                # get roi for subject with specific parameters
                get_rois(
                    sub_id,
                    t_area,
                    p_thresh
                )
            except FileNotFoundError:
                fail_count += 1
            print(f"{sub_id}: completed [{t_area}, {p_thresh}]")
        print(f"FINISHED {sub_id}")
    return fail_count


if __name__ == '__main__':
    # sample subject data
    # print(get_rois("sub-NSxLxYKx1964"))
    # get all rois using parameters in .env
    load_dotenv()
    sub_ids = json.loads(os.environ['SUB_IDS'])
    print(type(sub_ids))
    t_areas = json.loads(os.environ['T_AREAS'])
    p_value = json.loads(os.environ['P_THRESH'])
    print("CHECK VALUES")
    print(sub_ids)
    print(t_areas)
    print(p_value)
    time.sleep(5)
    print(batch_rois(sub_ids, t_areas, p_value))
