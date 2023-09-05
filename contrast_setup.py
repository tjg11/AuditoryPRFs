from design_matrix import find_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image, masking, plotting
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from nilearn.image import mean_img

plot_z_maps = True

data, event_matrix = find_design_matrix('sub-NSxLxIUx1994', '1')
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 6), nrows=1, ncols=3)
plot_design_matrix(event_matrix, ax=ax1)
ax1.set_title("Event-related design matrix", fontsize=12)
# plt.show()
print(data)

sub_id = 'sub-NSxLxIUx1994'
# base_path = os.path.abspath("OneDrive - UW/AMPB/data")
base_path = "../../AMPB/data"
anat_path = os.path.join(base_path,
                         sub_id,
                         'ses-01',
                         'func',
                         f'{sub_id}_ses-01_task-ptlocal_run-1_bold.nii.gz')
anat_data = nib.load(anat_path)
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


# z_map = fmri_glm.compute_contrast("stationary - silent")
# plotting.plot_stat_map(z_map, bg_img=mean_image, threshold=3.1)
# html = plotting.view_img(z_map, bg_img=mean_image, threshold=3.1)
# html.open_in_browser()
# plt.show()

print("Computing contrasts")

# observed_timeseries = masker.fit_transform(anat_data)
# predicted_timeseries = masker.fit_transform(fmri_glm.predicted[0])
if plot_z_maps:
    # Iterate on contrasts
    for contrast_id, contrast_val in contrasts.items():
        print(f"\tcontrast id: {contrast_id}")
        # compute the contrasts
        z_map = fmri_glm.compute_contrast(contrast_val, output_type="z_score")
        html = plotting.view_img(z_map, bg_img=mean_image, threshold=2.5)
        # html.open_in_browser()
        html.save_as_html(f"{contrast_id}.html")
        # plot the contrasts as soon as they're generated
        # the display is overlaid on the mean fMRI image
        # a threshold of 3.0 is used, more sophisticated choices are possible
        # plotting.plot_stat_map(
        #         z_map,
        #         bg_img=mean_image,
        #         threshold=2.0,
        #         display_mode="z",
        #         cut_coords=4,
        #         black_bg=True,
        #         title=contrast_id,
        #     )
        # plotting.show()
