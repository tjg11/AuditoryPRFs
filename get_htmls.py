from design_matrix import find_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import masking, plotting
import numpy as np
import os
import nibabel as nib
from nilearn.image import mean_img
import pickle


def get_htmls(sub_id,
              run_number,
              base_path,
              save_htmls=False,
              save_zmaps=False,
              save_nifti=False):

    file_name = f'{sub_id}_ses-01_task-ptlocal_run-{run_number}_bold.nii.gz'
    anat_path = os.path.join(base_path,
                             sub_id,
                             'ses-01',
                             'func',
                             file_name)
    if not os.path.exists(anat_path):
        print(f"Data not found for {anat_path}. Check path or subject data")
        return
    data, event_matrix = find_design_matrix(sub_id, run_number)
    print(data)

    # if not os.path.exists(save_path):
    #     os.makedirs("html_viewers")
    anat_data = nib.load(anat_path)
    mean_image = mean_img(anat_data)
    mask = masking.compute_epi_mask(mean_image)

    # Clean and smooth data
    # anat_data = image.clean_img(anat_data, standardize=False)
    # anat_data = image.smooth_img(anat_data, 5.0)

    contrast_matrix = np.eye(event_matrix.shape[1])
    b_con = {
        column: contrast_matrix[i]
        for i, column in enumerate(event_matrix.columns)
    }

    contrasts = {
        # "silent-sound":
        #  b_con["silent"] - (b_con["stationary"] + b_con["motion"]),
        "sound-silent":
        (b_con["stationary"] + b_con["motion"]) - b_con["silent"]
        # "silent-stationary":
        #  b_con["silent"] - b_con["stationary"],
        # "silent-motion":
        #  b_con["silent"] - b_con["motion"],
        # "effects_of_interest":
        #  np.vstack((b_con["silent"], b_con["motion"]))
    }

    fmri_glm = FirstLevelModel(
        drift_model="cosine",
        signal_scaling=False,
        mask_img=mask,
        minimize_memory=False,)
    print(anat_data.shape, event_matrix.shape)
    fmri_glm = fmri_glm.fit(anat_data, design_matrices=event_matrix)

    print("Computing contrasts")

    # Iterate on contrasts
    for contrast_id, contrast_val in contrasts.items():
        print(f"\tcontrast id: {contrast_id}")
        # compute the contrasts
        z_map = fmri_glm.compute_contrast(contrast_val, output_type="z_score")
        html = plotting.view_img(z_map, bg_img=mean_image, threshold=2.5)
        if save_htmls:
            file_name = f"{sub_id}_run{run_number}_{contrast_id}.html"
            file_name = os.path.join("html_viewers", sub_id, file_name)
            if not os.path.exists("html_viewers"):
                os.mkdir("html_viewers")
            if not os.path.exists(os.path.join("html_viewers", sub_id)):
                os.mkdir(os.path.join("html_viewers", sub_id))
            html.save_as_html(file_name)
        if save_zmaps:
            file_name = f"{sub_id}_run{run_number}_{contrast_id}.pickle"
            file_name = os.path.join("zmaps", sub_id, file_name)
            if not os.path.exists("zmaps"):
                os.mkdir("zmaps")
            if not os.path.exists(os.path.join("zmaps", sub_id)):
                os.mkdir(os.path.join("zmaps", sub_id))
            with open(file_name, "wb") as f:
                pickle.dump(z_map, f)

        if save_nifti:
            file_name = f"{sub_id}_run{run_number}_{contrast_id}.nii"
            file_name = os.path.join("niftis", sub_id, file_name)
            if not os.path.exists("niftis"):
                os.mkdir("niftis")
            if not os.path.exists(os.path.join("niftis", sub_id)):
                os.mkdir(os.path.join("niftis", sub_id))
            nib.save(z_map, file_name)
