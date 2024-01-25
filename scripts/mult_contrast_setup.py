from design_matrix import find_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image, masking, plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib

plot_z_maps = False

design_matricies = []
for run in range(1, 4):

    data, event_matrix1 = find_design_matrix('sub-NSxLxIUx1994', f'{run}')
    design_matricies.append(event_matrix1)



# plt.show()
print(data)

sub_id = 'sub-NSxLxIUx1994'
# base_path = os.path.abspath("OneDrive - UW/AMPB/data")
base_path = "../../AMPB/data"

anat_path1 = os.path.join(base_path, sub_id, 'ses-01', 'func', f'{sub_id}_ses-01_task-ptlocal_run-1_bold.nii.gz')
anat_data1 = nib.load(anat_path1)

anat_path2 = os.path.join(base_path, sub_id, 'ses-01', 'func', f'{sub_id}_ses-01_task-ptlocal_run-2_bold.nii.gz')
anat_data2 = nib.load(anat_path2)

anat_path3 = os.path.join(base_path, sub_id, 'ses-01', 'func', f'{sub_id}_ses-01_task-ptlocal_run-3_bold.nii.gz')
anat_data3 = nib.load(anat_path3)

anat_data = [anat_data1, anat_data2, anat_data3]
from nilearn.image import concat_imgs, mean_img, resample_img
mean_image = mean_img(anat_data[0])
mask = masking.compute_epi_mask(mean_image)

# Clean and smooth data
# anat_data = image.clean_img(anat_data, standardize=False)
# anat_data = image.smooth_img(anat_data, 5.0)

# contrast_matrix = np.eye(event_matrix.shape[1])
# basic_contrasts = {
# column: contrast_matrix[i]
# for i, column in enumerate(event_matrix.columns)
# }
    
# contrasts = {
#     "blank-sound": basic_contrasts["silent"] - (basic_contrasts["stationary"] + basic_contrasts["motion"]),
#     "sound-blank": (basic_contrasts["stationary"] + basic_contrasts["motion"]) - basic_contrasts["silent"],
#     "blank-stationary": basic_contrasts["silent"] - basic_contrasts["stationary"],
#     "blank-motion": basic_contrasts["silent"] - basic_contrasts["motion"],
#     "effects_of_interest": np.vstack((basic_contrasts["silent"], basic_contrasts["motion"]))
# }


fmri_glm = FirstLevelModel(
drift_model="cosine",
signal_scaling=False,
mask_img=mask,
minimize_memory=False,)
fmri_glm = fmri_glm.fit(anat_data, design_matrices=design_matricies)


z_map = fmri_glm.compute_contrast("motion - silent")
plotting.plot_stat_map(z_map, bg_img=mean_image, threshold=3.1)
html = plotting.view_img(z_map, bg_img=mean_image, threshold=3.1)
html.open_in_browser()
plt.show()

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
        html.open_in_browser()
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


