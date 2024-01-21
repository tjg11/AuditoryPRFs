# Auditory pRFs 

---

# Step 1: Create a stimulus record using stimulus_design.py

Creates a .pickle file from stimulus information contained in a .mat file.
Created file can be saved either in binary form or in index form. In order to
function correctly, the .mat file needs to have the followin structures:

- 'conditions'
- 'directions'

These two structures should describe the stimulus itself and the meaning of each
value in the stimulus, respectivley. The following parameters in settings.py may
also need to be changed depending on the stimulus type:

- 'S_LEN'
- 'TRIAL_LEN'
- 'RESOLUTION'
- 'DIR_LOC'
- 'SEL_COL'
- 'VAL_RIGHT'
- 'N_BURSTS'

By default, the settings correspond to the stimuli used in AMPB.

Note: The script 'batch_stimulus_design.py' can be used to create records for
multiple subjects with a single function call. Paths used in .env may need to
be altered for the batch script to work.

---

# Step 2: Convolve stimulus with hRF using convolve_hrf.py

Convolves a stimulus record with a hemodynamic response function (hRF). Returns
the convolved stimulus upon completion, and by setting 'save_pickle' to True,
also saves the convolved stimulus to a .pickle file. 

The stimulus record created in the previous step can be used here, although it
does not need to be a record created from the previous step. 

The specific parameters for the hRF can be changed by passing a custom
dictionary to 'hrf_params'. This dictionary must have the keys 'delta', 'tau', 
and 'n' in order to create an hRF. 

---

# Step 3 (Optional): Create ROI masks using get_rois_reworked.py

Creates custom region of interest (ROI) masks using information from software
like fMRIPrep. In order to create a mask, the function must be passed a loaded
segmentation image, as well as a corresponding index table containing the values
for the segmentation image. Additionally, it must be passed specific values to
filter for that exist within the index table.

The ROIs are further refined by using z-scores and p-values for each voxel.
These maps can be created using get_contrast_maps.py, which returns an image
for each previously mentioned statistic.

Lastly, the ROIs are defined and selected based on a certain size threshold,
where ROIs containing less voxels than specified in the threshold are not
retained, 

This function returns an array representing the ROI mask with the same
dimensions as the segmentation image, as well as a count of the number of voxels
in the mask. The script batch_get_rois.py can be used to create masks for
multiple subjects with various parameters for each ROI. 

---

# Step 4: Calculate pRFs using find_prf_updated.py

Explanation: Uses convolved stimuli files, ROI masks (if created) and model
parmeters defined in function call to calculate center, spread, and error values
for each voxel. Saves out a pickle file for each subject passed containing a
dictionary (with keys: mus, sigmas, error) where the values are

---

# Step 5 (optional): Plot pRF values

There are two scripts that can be used for visualizing the pRF results. The
first, hemi_split.py, reports the average values for each hemisphere. The
second, plot_results.py, displays a specific slice of the image overlaid with
one of the pRF result categories.

