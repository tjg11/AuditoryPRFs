import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation


def get_rois(
        seg_img: list,
        brain_mask: list,
        label_csv: list,
        labels: list,
        p_values: list,
        z_scores: list,
        p_thresh,
        roi_size
):
    """
    Takes one loaded MRI image, the corresponding segmentation file, the
    target areas from that segmentation file to be masked for, a map of
    p-values for thresholding, and a map of z-scores for thresholding (both
    NEED to be the same size as the loaded image). Returns the array containing
    the ROI mask, and the number of voxels contained in the mask, respecitvley.
    """

    # check that shape of maps matches shape of image
    if seg_img.shape != p_values.shape:
        print("Shape of image and p-values does not match")
        return
    if seg_img.shape != z_scores.shape:
        print("Shape of image and z-scores does not match")
        return

    # rename problematic column name if present in label_csv
    if "index" in label_csv.columns:
        label_csv = label_csv.rename(
            columns={
                'index': 'label_index'
            }
        )

    # find indicies to filter for in seg_img
    target_idxs = []
    for label_id in labels:
        target = label_csv[label_csv.name == label_id]
        target_idx = target.iloc[0][0]
        target_idxs.append(target_idx)

    # create mask array using target indicies
    for target_idx in target_idxs:
        filt_seg = seg_img.copy()
        filt_seg[filt_seg == target_idx] = 0
        seg_img = filt_seg * seg_img

    # flip values so that 1 means keep and 0 means don't keep
    seg_img[seg_img != 0] = 1
    seg_img = 1 - seg_img

    # threshold z-map using p-values and binarize
    z_scores[p_values > p_thresh] = 0
    bin_z_scores = z_scores != 0

    # create and filter labels based on target area
    label_image = label(bin_z_scores, connectivity=1)
    for region in regionprops(label_image):
        if region.area < roi_size:
            label_image[label_image == region.label] = 0

    # binarize label image
    bin_label_image = label_image != 0

    # dilate label image
    bin_label_image = binary_dilation(bin_label_image)

    # binarize brain mask
    bin_brain_mask = brain_mask != 0

    # apply brain mask to dilated label image
    brain_label_image = bin_label_image * bin_brain_mask

    # apply seg_img image to label image
    final_image = brain_label_image * seg_img

    # get count of included voxels
    voxels = np.count_nonzero(final_image)

    # return mask and voxel count
    return final_image, voxels
