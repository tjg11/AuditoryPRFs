# Author: Taylor Garrison
# Description: Designed to convolve a hemodynamic
# response function (HRF) with a stimulus
# created from the stimulus_creation.py script in this respository.
#  Convolved stimulus used in following analysis. Saved out as a binary file.

import scipy
import numpy as np
import pickle
from os import path as op


def convolve_hrf(
    subject_id,
    run_number,
    hrf_params={
        "delta": 2.25,
        "tau": 1.2,
        "n": 3
    }
):

    dt = 0.01

    # define HRF parameters
    delta = hrf_params["delta"]
    tau = hrf_params["tau"]
    n = hrf_params["n"]

    #TODO: pre- and post- convolution interpolating, under assumption that the 
    # stimulus image is created at the same time resolution
    th = np.arange(0, 30, dt)

    t_delta = th - delta  # shift time vector for lag
    h_exp = np.exp(-t_delta / tau) / (tau * np.math.factorial(n - 1))
    h = ((t_delta / tau) ** (n - 1)) * h_exp
    h[th < delta] = 0  # remove values prior to delta lag

    # Open binarized data file from stimulus_creation.py
    f_name = op.join(subject_id, f"binary_{subject_id}_{run_number}.pickle")
    with open(f_name, "rb") as f:
        S = pickle.load(f)

    # Print shape and plot data for sanity check
    print(f"Shape of subject data: {S.shape}")

    C = np.apply_along_axis(
        lambda x: scipy.signal.convolve(x, h, mode="same"),
        axis=0,
        arr=S
    )

    #TODO: if save_pickle
    print(C.shape)
    img_name = f"convolved_{subject_id}_{run_number}.pickle"
    img_name = op.join(subject_id, img_name)
    with open(img_name, "wb") as f:
        pickle.dump(C, f)
    
    #TODO: return the convolved stimulus

