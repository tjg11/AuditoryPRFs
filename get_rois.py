import pickle
import os
from nilearn.image import mean_img, concat_imgs
import nibabel as nib
from nilearn import plotting
import numpy as np

sub_id = "sub-NSxLxIUx1994"
run_number = "1"
file_name = os.path.join("zmaps",
                         sub_id,
                         f"{sub_id}_run{run_number}_sound-silent.pickle")
base_path = "../../AMPB/data"
anat_path = f'{sub_id}_ses-01_task-ptlocal_run-{run_number}_bold.nii.gz'
anat_path = os.path.join(base_path,
                         sub_id, 'ses-01',
                         'func',
                         anat_path)
anat_data = nib.load(anat_path)
mean_image = mean_img(anat_data)
with open(file_name, "rb") as f:
    z_map = pickle.load(f)

print(z_map.shape)

x_coord = -49
y_coord = 20
z_coord = 27

file_name = os.path.join("niftis",
                         sub_id,
                         f"{sub_id}_run{run_number}_sound-silent.nii")

file_name1 = os.path.join("niftis",
                          sub_id,
                          f"{sub_id}_run1_sound-silent.nii")

file_name2 = os.path.join("niftis",
                          sub_id,
                          f"{sub_id}_run2_sound-silent.nii")

file_name3 = os.path.join("niftis",
                          sub_id,
                          f"{sub_id}_run3_sound-silent.nii")

all_zmaps = concat_imgs([file_name1, file_name2, file_name3])


z_map = nib.load(file_name)
z_data = z_map.get_fdata()
print(z_data[x_coord][y_coord][z_coord])
print(z_data.min())

max_idx = np.unravel_index(np.argmin(z_data), z_data.shape)

plotting.plot_stat_map(
    all_zmaps,
    bg_img=mean_image,
    threshold=3,
    display_mode="z",
)

plotting.show()

print(max_idx, z_data[max_idx])

# regions_value_img, index = connected_regions(
#     z_map, min_region_size=200
# )
# plotting.plot_stat_map(
#                 z_map,
#                 bg_img=mean_image,
#                 threshold=2.0,
#                 display_mode="z",
#                 cut_coords=4,

#              )
# plotting.plot_prob_atlas(
#     regions_value_img,
#     bg_img=mean_image,
#     view_type="contours",
#     display_mode="z",
#     cut_coords=5,
#     )
# plotting.show()
