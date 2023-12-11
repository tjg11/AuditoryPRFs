from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import masking, plotting
import numpy as np
import os
import json
import nibabel as nib
from nilearn.image import mean_img
from dotenv import load_dotenv

def design_matrix(sample_rate: int,
                  n_volumes: int,
                  events):
    """
    Returns a first level design matrix using sampling rate, number of volumes
    and the correpsonding event timings of a functional scan. Events should be
    in a dataframe. Returns the design matrix
    """
    # create time vector (timing of scan acquisition)
    time_vector = np.arange(n_volumes * sample_rate)
    # create the design matrix
    design_matrix = make_first_level_design_matrix(time_vector, events)
    # return the design matrix
    return design_matrix

def get_contrast_maps(data: list,
                      design_matricies: list,
                      contrast_labels: list,
                      constrast_categories: list)