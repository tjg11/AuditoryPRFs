import os
import pickle
import time
import json
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation

def get_rois(
        img: list,

):
    """
    Takes one loaded MRI image, the corresponding segmentation file, the
    target areas from that segmentation file to be masked for, a map of p-values
    for thresholding, and a map of z-scores for thresholding (both NEED to be
    the same size as the loaded image). Returns the array containing the
    ROI mask, and the number of voxels contained in the mask, respecitvley.
    """