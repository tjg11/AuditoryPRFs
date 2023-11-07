from design_matrix import find_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import masking, plotting
import numpy as np
import os
import nibabel as nib
from nilearn.image import mean_img


def get_htmls(sub_id,
              num_runs,
              base_path,
              save_htmls=False,
              save_zmaps=False,
              save_nifti=False,
              save_bg_image=False):
    fmri_img = []
    design_matricies = []
    for run in range(1, num_runs + 1):
        r_num = str(run)
        file_name = f'{sub_id}_ses-01_task-ptlocal_run-{r_num}_bold.nii.gz'
        anat_path = os.path.join(base_path,
                                 sub_id,
                                 'ses-01',
                                 'func',
                                 file_name)
        if not os.path.exists(anat_path):
            print(
                f"Data not found for {anat_path}. Check path or subject data")
            return
        data = nib.load(anat_path)
        fmri_img.append(data)
        data, event_matrix = find_design_matrix(sub_id, r_num)
        design_matricies.append(event_matrix)

    # if not os.path.exists(save_path):
    #     os.makedirs("html_viewers")
    mean_image = mean_img(fmri_img)
    mask = masking.compute_epi_mask(mean_image)

    contrast_matrix = np.eye(event_matrix.shape[1])
    b_con = {
        column: contrast_matrix[i]
        for i, column in enumerate(event_matrix.columns)
    }

    contrasts = {
        "silent-sound":
        b_con["silent"] - (b_con["stationary"] + b_con["motion"]),
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

    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matricies)

    print("Computing contrasts")

    # Iterate on contrasts
    for contrast_id, contrast_val in contrasts.items():
        print(f"\tcontrast id: {contrast_id}")
        # compute the contrasts
        z_map = fmri_glm.compute_contrast(contrast_val,
                                          output_type="z_score")
        p_values = fmri_glm.compute_contrast(contrast_val,
                                             output_type="p_value")
        html = plotting.view_img(z_map, bg_img=mean_image, threshold=2.5)
        if save_htmls:
            file_name = f"{sub_id}_{contrast_id}.html"
            file_name = os.path.join("html_viewers", sub_id, file_name)
            if not os.path.exists("html_viewers"):
                os.mkdir("html_viewers")
            if not os.path.exists(os.path.join("html_viewers", sub_id)):
                os.mkdir(os.path.join("html_viewers", sub_id))
            html.save_as_html(file_name)
        if save_zmaps:
            file_name = f"{sub_id}_{contrast_id}.nii"
            file_name = os.path.join("zmaps", sub_id, file_name)
            if not os.path.exists("zmaps"):
                os.mkdir("zmaps")
            if not os.path.exists(os.path.join("zmaps", sub_id)):
                os.mkdir(os.path.join("zmaps", sub_id))
            nib.save(z_map, file_name)

        if save_nifti:
            file_name = f"{sub_id}_{contrast_id}.nii"
            file_name = os.path.join("niftis", sub_id, file_name)
            if not os.path.exists("niftis"):
                os.mkdir("niftis")
            if not os.path.exists(os.path.join("niftis", sub_id)):
                os.mkdir(os.path.join("niftis", sub_id))
            nib.save(p_values, file_name)

        if save_bg_image:
            if not os.path.exists("mean_img"):
                os.mkdir("mean_img")
            file_name = os.path.join("mean_img", f"{sub_id}_mean_img.nii")
            nib.save(mean_image, file_name)


if __name__ == '__main__':
    where = os.path.abspath(os.path.join("..", "..", "AMPB", "data"))
    get_htmls('sub-NSxLxYKx1964', 3, where, save_zmaps=True)
