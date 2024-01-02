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

    # get initial number of voxel below threshold
    p_cnt = np.count_nonzero(p_values > p_thresh)
    print(f"P-VALUES BELOW THRESHOLD: {p_cnt}")

    # find indicies to filter for in seg_img
    bin_seg = seg_img.copy()
    target_idxs = []
    for label_id in labels:
        target = label_csv[label_csv.name == label_id]
        target_idx = target.iloc[0][0]
        target_idxs.append(target_idx)

    # create mask array using target indicies & keep track of matching voxels
    m_cnt = 0
    for target_idx in target_idxs:
        print(F"UNIQUE VALUES IN COPIED SEGMENTATION: {np.unique(bin_seg)}")
        m_cnt += np.count_nonzero(bin_seg == target_idx)
        print(F"VOXELS IN TARGET {target_idx}: {m_cnt}")
        bin_seg[bin_seg == target_idx] = -1

    # flip values so that 1 means keep and 0 means don't keep
    bin_seg[bin_seg != -1] = 0
    bin_seg[bin_seg == -1] = 1

    # get number of voxels in segmentation mask
    s_cnt = np.count_nonzero(bin_seg)
    print(F"VOXELS IN SEGMENTATION MASK: {s_cnt}")
    print(f"MATCHING VOXELS IN ORIGINAL SEGMENTATION: {m_cnt}")
    print(F"UNIQUE VALUES IN SEGMENGATION MASK: {np.unique(bin_seg)}")

    # check value against number of matching indicies in

    # threshold z-map using p-values and binarize
    z_thresh_scores = z_scores.copy()
    z_thresh_scores[p_values > p_thresh] = 0
    bin_z_scores = z_thresh_scores != 0

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

    # apply bin_seg image to label image
    final_image = brain_label_image * bin_seg

    # get count of included voxels
    voxels = np.count_nonzero(final_image)

    # return mask and voxel count
    return final_image, voxels
