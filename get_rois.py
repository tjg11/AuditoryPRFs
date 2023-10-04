import os
import pickle
import difflib
from os import path as op
from nilearn.image import resample_img
import nibabel as nib
import numpy as np
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

    # load z-maps with original data
    f_name = op.join(paths_data_zmaps,
                     f"{sub_id}_sound-silent.nii")
    z_maps = nib.load(f_name).get_fdata()
    o_data = nib.load(f_name)  # non-loaded data for resampling params

    # load brain mask and resample
    file_text = "ses-01_acq-MEMPRvNav_rec-RMS_desc-brain_mask.nii.gz"
    f_name = op.join(paths_sub,
                     f"{sub_id}_{file_text}")
    if op.exists(f_name):
        print("Brain mask file exists!")
    else:
        print(f"\n{f_name}")
        new_base = op.abspath(os.path.join("..", "..", "AMPB", "data"))
        print(f"\n{new_base}")
        paths_sub = op.join(new_base,
                            "derivatives",
                            "fmriprep",
                            sub_id,
                            "ses-01",
                            "anat")
        new_f_name = op.join(paths_sub, f"{sub_id}_{file_text}")
        print(f"\n{new_f_name}")
        print(op.exists(new_f_name))
        print("Brain mask file not found!")
        print(len(f_name))
        print(len(new_f_name))
        print("are the two paths the same???")
        print(f_name)
        print(new_f_name)
        print(f"{new_f_name == f_name}")
        print(repr(new_f_name))
        print('\n'.join(difflib.ndiff([f_name], [new_f_name])))
        return
    ob_mask = nib.load(f_name)  # original mask
    rb_mask = resample_img(ob_mask,
                           target_affine=o_data.affine,
                           target_shape=o_data.shape)  # resampled b_mask
    bin_rb_mask = rb_mask != 0  # binarized resampled b_mask

    # load p-values
    f_name = op.join(paths_data_pvalue,
                     f"{sub_id}_sound-silent.nii")
    p_values = nib.load(f_name).get_fdata()

    # threshold z-map using p-values and binarize
    z_maps[p_values > p_threshold] = 0
    bin_zmaps = z_maps != 0

    # create  and filter labels based on target area
    label_image = label(bin_zmaps, connectivity=1)
    for region in regionprops(label_image):
        if region.area < target_area:
            label_image[label_image == region.label] = 0

    # binarize label image, dilate, and apply brain mask
    bin_label_image = label_image != 0
    dilated_label_image = binary_dilation(bin_label_image)
    final_label_image = dilated_label_image * bin_rb_mask
    vox_count = np.count_nonzero(final_label_image)
    print(f"Subject [{sub_id}] complete, see voxel count below.")
    print(f"The number of voxels in the final label image: {vox_count} ")

    # save files
    if save_file:
        f_name = op.join("rois",
                         f"{sub_id}_auditory_roi.pickle")
    if not op.exists("rois"):
        os.mkdir("rois")
    with open(f_name, "wb") as f:
        pickle.dump(final_label_image, f)
    return f"Finished [{sub_id}] successfully\n\n"


if __name__ == '__main__':
    print(get_rois("sub-NSxLxYKx1964"))
