from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image, masking
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib


def find_design_matrix(sub_id, run_number):

    # Sampling rate
    tr = 2
    n_volumes = 163
    time_vector = np.arange(n_volumes) * tr

    # base_path = os.path.abspath("OneDrive - UW/AMPB/data/")
    # TODO: Change this to be interchangeable
    base_path = os.path.join(
        "C:\\",
        "Users",
        "Taylor Garrison",
        "OneDrive - UW",
        "AMPB",
        "data"
    )
    data_name = f'{sub_id}_ses-01_task-ptlocal_run-{run_number}_events.tsv'
    data_path = os.path.join(base_path,
                             sub_id,
                             'ses-01',
                             'func',
                             data_name)
    event_data = pd.read_csv(data_path, sep='\t')

    X1 = make_first_level_design_matrix(time_vector, event_data)

    return event_data, X1


if __name__ == '__main__':
    data, event_matrix = find_design_matrix('sub-NSxLxIUx1994', '1')
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 6), nrows=1, ncols=3)
    plot_design_matrix(event_matrix, ax=ax1)
    ax1.set_title("Event-related design matrix", fontsize=12)
    plt.show()
    print(data)

    sub_id = 'sub-NSxLxIUx1994'
    # base_path = os.path.abspath("../../AMPB/data")
    base_path = "../../AMPB/data"
    anat_path = os.path.join(base_path,
                             sub_id,
                             'ses-01',
                             'func',
                             f'{sub_id}_ses-01_task-ptlocal_run-1_bold.nii.gz')
    anat_data = nib.load(anat_path)
    from nilearn.image import mean_img
    mean_image = mean_img(anat_data)
    mask = masking.compute_epi_mask(mean_image)

    # Clean and smooth data
    anat_data = image.clean_img(anat_data, standardize=False)
    anat_data = image.smooth_img(anat_data, 5.0)

    contrast_matrix = np.eye(event_matrix.shape[1])
    basic_contrasts = {
                        column: contrast_matrix[i]
                        for i, column in enumerate(event_matrix.columns)
                      }
    silent = basic_contrasts["silent"]
    motion = basic_contrasts["motion"]
    stationary = basic_contrasts["stationary"]
    sound = motion + stationary
    contrasts = {
                "blank-sound": silent - (sound),
                "sound-blank": (sound) - silent,
                "blank-stationary": silent - stationary,
                "blank-motion": silent - motion,
                "effects_of_interest": np.vstack((silent, motion))
                }

    fmri_glm = FirstLevelModel(
                                drift_model="cosine",
                                signal_scaling=False,
                                mask_img=mask,
                                minimize_memory=False,)
    print(anat_data.shape, event_matrix.shape)
    fmri_glm = fmri_glm.fit(anat_data, design_matrices=event_matrix)

    from nilearn import plotting

    z_map = fmri_glm.compute_contrast("silent - motion")

    plotting.plot_stat_map(z_map, bg_img=mean_image, threshold=3.1)

    print("Computing contrasts")

    # observed_timeseries = masker.fit_transform(anat_data)
    # predicted_timeseries = masker.fit_transform(fmri_glm.predicted[0])

    # Iterate on contrasts
    for contrast_id, contrast_val in contrasts.items():
        print(f"\tcontrast id: {contrast_id}")
        # compute the contrasts
        z_map = fmri_glm.compute_contrast(contrast_val, output_type="z_score")
        # plot the contrasts as soon as they're generated
        # the display is overlaid on the mean fMRI image
        # a threshold of 3.0 is used, more sophisticated choices are possible
        plotting.plot_stat_map(
            z_map,
            bg_img=mean_image,
            threshold=2.0,
            display_mode="z",
            cut_coords=4,
            black_bg=True,
            title=contrast_id,
        )
        plotting.show()
