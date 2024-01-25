from get_contrast_maps import get_contrast_maps, design_matrix
from dotenv import load_dotenv
import os
import json
import pandas as pd
import nibabel as nib
from glob import glob
from time import sleep


def batch_contrast_maps(
        sub_id: str,
        trials: list,
        scans: list,
        sample_rate: int,
        n_volumes: int,
        paths_save: str
        ):
    """
    Takes a subject identifier, a list containing paths to the .csv files for
    each trial, and a list containing paths to the scan files for each trial.
    Saves the resulting p-value and z-score contrast maps into their own
    subdirectories within the directory corresponding to the subject
    identifier. Returns the number of files saved.
    """

    # track files saved
    file_count = 0

    # get the design matrices for each trial
    design_matrices = []
    for trial in trials:
        # load each trial into a dataframe
        df = pd.read_csv(trial, sep='\t')
        df = design_matrix(sample_rate, n_volumes, df)
        design_matrices.append(df)

    # get contrast maps for subject
    z_map, p_map, contrast_id = get_contrast_maps(scans, design_matrices)

    # save z-scores
    file_name = f"{sub_id}_zscores_{contrast_id}.nii.gz"
    nib.save(z_map, os.path.join(paths_save, file_name))
    file_count += 1

    # save p-values
    file_name = f"{sub_id}_pvalues_{contrast_id}.nii.gz"
    nib.save(p_map, os.path.join(paths_save, file_name))
    file_count += 1

    return file_count


if __name__ == '__main__':

    # load environment variables
    load_dotenv()

    # get subject ids
    sub_ids = json.loads(os.getenv("SUB_IDS"))
    print(f"Number of subjects: {len(sub_ids)}.")
    print(f"First subject: {sub_ids[0]}.")

    # get save path
    path_save = os.getenv("DATA_PATH")

    # get experiment data path
    path_data = os.getenv("ORIG_PATH")
    path_scans = os.path.join(
        path_data,
        "derivatives",
        "fmriprep"
        )
    path_events = path_data

    # set experiment specific parameters
    sample_rate = 2
    n_volumes = 163

    # get contrast maps for each subject
    for sub_id in sub_ids:

        # get event files
        events = glob(
            os.path.join(
                path_events,
                sub_id,
                "ses-01",
                "func",
                "*ptlocal*.tsv"
            )
        )
        # check that all event files were found
        if len(events) != 3:
            print(f"Incorrect number of event files found for {sub_id}.")
            sleep(10)

        # get scan files
        scans = glob(
            os.path.join(
                path_scans,
                sub_id,
                "ses-01",
                "func",
                "*ptlocal*space-T1w_desc-preproc_bold.nii.gz"
            )
        )
        # check that all scan iles were found
        if len(scans) != 3:
            print(f"Incorrect number of scan files found for {sub_id}.")
            sleep(10)

        # load scan data
        fmri_img = []
        for scan in scans:
            img = nib.load(scan)
            fmri_img.append(img)

        # create save path
        save_loc = os.path.join(
            path_save,
            sub_id,
            "contrast_maps"
        )
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
            print(f"Directory created for {sub_id}.")

        # get contrast maps
        files_created = batch_contrast_maps(
            sub_id,
            events,
            fmri_img,
            sample_rate=2,
            n_volumes=163,
            paths_save=save_loc
        )

        # print number of files created
        print(f"Files created: {files_created} - {sub_id}.")
