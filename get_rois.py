import os
import pickle
from os import path as op
from nilearn.image import resample_img
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation
import warnings


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

    warnings.filterwarnings("ignore")

    # set base, main, and sub paths
    if paths_base is None:
        paths_base = op.join("C:\\",
                             "Users",
                             "Taylor Garrison",
                             "OneDrive - UW")
    if paths_main is None:
        paths_main = op.join(paths_base,
                             "Scripts",
                             "PythonScripts")
    if paths_sub is None:
        paths_sub = op.join(paths_base,
                            "AMPB",
                            "data",
                            "derivatives",
                            "fmriprep",
                            sub_id,
                            "ses-01",
                            "anat")

    # set data paths
    paths_data_zmaps = op.join(paths_main, "zmaps", sub_id)
    paths_data_pvalue = op.join(paths_main, "niftis", sub_id)

    # load z-maps with original data and check for file not found
    f_name = op.join(paths_data_zmaps,
                     f"{sub_id}_sound-silent.nii")
    if op.exists(f_name):
        print("Z-map file exists!")
    else:
        print("Z-map file doesn't exist! Check the path for errors.")
        return
    z_maps = nib.load(f_name).get_fdata()
    o_data = nib.load(f_name)  # non-loaded data for resampling params

    # load brain mask and resample and check for file not found
    file_text = "ses-01_acq-MEMPRvNav_rec-RMS_desc-brain_mask.nii.gz"
    f_name = op.join(paths_sub,
                     f"{sub_id}_{file_text}")
    if op.exists(f_name):
        print("Brain mask file exists!")
    else:
        print("Brain mask file doesn't exist! Check the path for errors.")
        return
    ob_mask = nib.load(f_name)  # original mask
    rb_mask = resample_img(ob_mask,
                           target_affine=o_data.affine,
                           target_shape=o_data.shape)  # resampled b_mask
    bin_rb_mask = rb_mask != 0  # binarized resampled b_mask

    # load segmentation and resample
    file_text = f"{sub_id}_ses-01_acq-MEMPRvNav_rec-RMS_desc-aseg_dseg.nii.gz"
    f_name = op.join(paths_sub,
                     file_text)
    if op.exists(f_name):
        print("Segmentation file exists!")
    else:
        print("Segmentation file does not exist. Check for path errors.")
    oseg = nib.load(f_name)
    rseg = resample_img(
        oseg,
        target_affine=o_data.affine,
        target_shape=o_data.shape
    )
    rseg = rseg.get_fdata().astype(int)

    # load segmentation legend and find target indicies based on seg_areas
    f_name = op.join(paths_base,
                     "AMPB",
                     "data",
                     "derivatives",
                     "fmriprep",
                     "desc-aseg_dseg.tsv")
    legend = pd.read_csv(f_name, sep='\t')
    legend = legend.rename(
        columns={
            'index': 'legend_index'
        })  # rename column b/c index is a keyword
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
                     f"{sub_id}_sound-silent.nii")
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
        f_name = op.join("rois",
                         sub_id,
                         file_text)
    if not op.exists("rois"):
        os.mkdir("rois")
    if not op.exists(op.join("rois", sub_id)):
        os.mkdir(op.join("rois", sub_id))
    with open(f_name, "wb") as f:
        pickle.dump(final_label_image, f)
    return f"Finished [{sub_id}] successfully\n\n"


if __name__ == '__main__':
    print(get_rois("sub-NSxLxYKx1964"))
