import scipy.io
import os
import numpy as np
import nibabel as nib
import pickle
import pandas as pd
import json
import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix


def open_matfile(path):
    file = scipy.io.loadmat(path)
    conditions = file['conditions'][0][0][0]
    c_matrix = np.matrix(np.array(conditions))
    return c_matrix.shape


def open_created_record(path):
    with open(path, "rb") as f:
        record = pickle.load(f)
    return record.shape


def record_from_tsv(path, rep_time, volumes):
    dt = np.arange(volumes) * rep_time
    df = pd.read_csv(path, sep='\t')
    design_matrix = make_first_level_design_matrix(
        dt,
        df
    )
    return design_matrix


def get_time_params(path):
    with open(path) as f:
        scan = json.load(f)
    tr = scan["RepetitionTime"]
    return tr


def get_shape(path):
    img = nib.load(path).get_fdata()
    return img.shape


if __name__ == '__main__':

    path = os.path.join(
        "C:\\",
        "Users",
        "Taylor Garrison",
        "OneDrive - UW",
        "AMPB",
        "data",
        "sub-NSxLxYKx1964",
        "ses-02",
        "func",
        "sub-NSxLxYKx1964_ses-02_task-AMPB_run-1_20220815_1100.mat"
    )

    path2 = os.path.join(
        "C:\\",
        "Users",
        "Taylor Garrison",
        "OneDrive - UW",
        "Scripts",
        "PythonScripts",
        "sub-NSxLxYKx1964",
        "binary_sub-NSxLxYKx1964_01.pickle"
    )

    path3 = os.path.join(
        "C:\\",
        "Users",
        "Taylor Garrison",
        "OneDrive - UW",
        "AMPB",
        "data",
        "derivatives",
        "fmriprep",
        "sub-NSxLxYKx1964",
        "ses-02",
        "func",
        "sub-NSxLxYKx1964_ses-02_task-ampb_run-1_space-T1w_desc-preproc_bold.json"
    )

    path4 = os.path.join(
        "C:\\",
        "Users",
        "Taylor Garrison",
        "OneDrive - UW",
        "AMPB",
        "data",
        "derivatives",
        "fmriprep",
        "sub-NSxLxYKx1964",
        "ses-02",
        "func",
        "sub-NSxLxYKx1964_ses-02_task-ampb_run-1_space-T1w_desc-preproc_bold.nii.gz"
    )
    
    path5 = os.path.join(
        "C:\\",
        "Users",
        "Taylor Garrison",
        "OneDrive - UW",
        "AMPB",
        "data",
        "sub-NSxLxYKx1964",
        "ses-02",
        "func",
        "sub-NSxLxYKx1964_ses-02_task-ampb_run-1_events.tsv"
    )

    print(open_matfile(path))
    print(open_created_record(path2))
    print(get_time_params(path3))
    print(get_shape(path4))
    record = record_from_tsv(path5, 2, 215)
    print(type(record))
    print(record.shape)
    fig, ax = plt.subplots()
    plot_design_matrix(record, ax=ax)
    plt.show()
