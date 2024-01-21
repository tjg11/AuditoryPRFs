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

Explanation: Uses files created by fMRIPrep (e.g. label csv file, label mask) to
create specific ROIs based on the labels passed to the function. Can use p-values
and z-scores from get_contrast_maps.py to further refine each ROI. Saves out each
ROI mask as a nifti file.

---

# Step 4: Calculate pRFs using find_prf_updated.py

Explanation: Uses convolved stimuli files, ROI masks (if created) and model
parmeters defined in function call to calculate center, spread, and error values
for each voxel. Saves out a pickle file for each subject passed containing a
dictionary (with keys: mus, sigmas, error) where the values are

---

# Step 5 (optional): Plot pRF values using scripts in plotting subdirectory