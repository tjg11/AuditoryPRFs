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
                      design_matricies: list):
    """
    Takes an array of scans and a corrsepdong array of design matricies
    and calculates contrast maps using motion labels. ONLY WORKS with design
    matrix labels "silent", "stationary", and "motion". Returns z-score map,
    p-value map, and contrast lable, respectivley, comparing sound conditions
    to silent conditions.
    """

    # create contrast map based on shape of single design matrix
    sample_matrix = design_matricies[0]
    contrast_matrix = np.eye(sample_matrix.shape[1])

    # create binary contrast matrix for create actual contrast labels
    b_con = {
        column: contrast_matrix[i]
        for i, column in enumerate(sample_matrix.columns)
    }

    # create contrast label matrix
    contrasts = {
        "sound-silent":
        (b_con["stationary"] + b_con["motion"]) - b_con["silent"]
    }

    # perform first level model fit
    fmri_glm = FirstLevelModel(
        drift_model="cosine",
        signal_scaling=False,
        minimize_memory=False
    )
    fmri_glm = fmri_glm.fit(data, design_matricies=design_matricies)

    # calculate contrasts using label matrix and glm fit
    for contrast_id, contrast_val in contrasts.items():
        z_map = fmri_glm.compute_contrast(
            contrast_val,
            output_type="z_score"
        )
        p_values = fmri_glm.compute_contrsast(
            contrast_val,
            output_type="p_value"
        )

        return z_map, p_values, contrast_id
