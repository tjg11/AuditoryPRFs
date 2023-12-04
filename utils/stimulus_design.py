import os
import glob
import scipy.io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from settings import (S_LEN,
                      TRIAL_LEN,
                      RESOLUTION,
                      DIR_LOC,
                      DIST,
                      SEL_COL,
                      VAL_RIGHT,
                      N_BURSTS
                      )


def get_matfiles(path: str) -> list:
    """Returns list of matfiles in subject directory. Takes path to subject
    as parameter."""

    # set path
    search_path = os.path.join(
        path,
        "*.mat"
    )

    # get matfiles
    matfiles = glob.glob(search_path)
    return matfiles


def get_conditions(path: str) -> list:
    """Returns array of conditions based on matfile. Takes path to matfile
    as parameter."""
    # load matfile as dict with variables as keys
    mat_file = scipy.io.loadmat(path)
    # try to get conditions out of dict
    try:
        conditions = mat_file['conditions'][0][0][0]
    except KeyError:
        print("Conditions not found in matfile.")
        return []
    except IndexError:
        print("Error in indexing matfile.")
        return []
    finally:
        return conditions


def get_directions(conditions: list) -> tuple:
    """Returns tuple containing directions found in conditions array
    and conditions themselves, respectivley. Returns empty lists if index
    error occurred."""
    try:
        directions = conditions[:, DIR_LOC]
    except IndexError:
        print("Directions not found based on index.")
        directions = []
    try:
        conditions = conditions[:, SEL_COL]
    except IndexError:
        print("Actual conditions not found based on index.")
        conditions = []
    return directions, conditions


def make_empty_record() -> list:
    """Returns an empty record with size of
    ([TRIAL_LEN / RESOLUTION], S_LEN)."""
    # set number of bins
    n_bins = TRIAL_LEN / RESOLUTION
    return np.zeros((int(n_bins), S_LEN))


def save_record(stim_record: list,
                label: str,
                out_path: str,
                options: str) -> int:
    """Saves stimulus record in specified formats based on parameters in
    stimulus_creation. Returns 0 if saved successfully, otherwise returns 1."""
    # save as fig
    if options == 'fig':
        fname = os.path.join(out_path, f"stim_record{label}.png")
        plt.imsave(fname, stim_record)
    # save as bin
    if options == 'bin':
        fname = os.path.join(out_path, f"stim_record{label}.pickle")
        with open(fname, "wb") as f:
            pickle.dump(stim_record, f)


def mark_sound_location(target_record: list,
                        target_val: int,
                        target_row: int,
                        flipped: int) -> None:
    """Helper function for make_val_stim_record. Uses parameters to record
    sounds location in stimulus record. Returns none."""
    target_idx = int(target_val - 1)
    # get location in distance space
    dist = DIST[target_idx]
    # change index if flipped
    if flipped == -1:
        target_idx = (abs(target_idx - S_LEN)) - 1
    target_record[target_row, target_idx] = dist * flipped
    return None


def mark_bin_location(
        target_record: list,
        target_val: int,
        target_row: int,
        flipped: int) -> None:
    """Helper function for make_bin_stim_record. Uses parameters to record
    sounds location in stimulus record. Returns none."""
    target_idx = int(target_val - 1)
    # change index if flipped
    if flipped == -1:
        target_idx = (abs(target_idx - S_LEN)) - 1
    # convert to sound location TODO: change hardcoding
    target_idx = target_idx * 7.5
    target_record[target_row, target_idx] = 1
    return None


def make_val_stim_record(
        empty_record: list,
        directions: list,
        conditions: list) -> list:
    """Fills in stim_record based on stimulus in matfile and parameters in
    settings. Returns complete record."""
    # set tracking variables
    stim_record = empty_record.copy()
    sample_x = 0
    sample_y = 0
    times_rep = 0
    n_burst = 0
    max_reps = S_LEN + 1
    # iterate through rows of empty record and fill in values
    for sr_row in range(stim_record.shape[0]):
        # get directions
        flipped = 1
        if directions[sample_y] == VAL_RIGHT:
            flipped = -1
        # repeat until reaching max_reps and then don't change values
        if times_rep == max_reps:
            times_rep += 1
        # insert location of audio
        elif times_rep < max_reps:
            # if still on sound
            if sample_x < S_LEN:
                # get location
                target = conditions[sample_y, sample_x]
                # add if bursts less than N_BURSTS
                if n_burst < N_BURSTS:
                    mark_sound_location(stim_record, target, sr_row, flipped)
                    n_burst += 1
                # if bursts complete, increase x_sample and reset bursts
                else:
                    sample_x += 1
                    n_burst = 0
                # mark first location of next burst if not the last sound
                    if sample_x < 9 and times_rep != max_reps:
                        n_burst = 1
                        mark_sound_location(stim_record,
                                            target,
                                            sr_row,
                                            flipped)
                    elif times_rep != max_reps - 1:
                        n_burst = 1
                        target = conditions[sample_y, 0]
                        mark_sound_location(stim_record,
                                            target,
                                            sr_row,
                                            flipped)
                    else:
                        times_rep += 1
            # increase repreat counter and reset sample x index
            else:
                times_rep += 1
                sample_x = 0
                if times_rep != max_reps:
                    n_burst += 1
                    mark_sound_location(stim_record,
                                        target,
                                        sr_row,
                                        flipped)
                else:
                    n_burst = 0
        # plot blank space until reaching trial time
        elif (sr_row % 1600) != 1599:
            times_rep += 1
        # move on to next block of sounds
        else:
            sample_y += 1
            times_rep = 0
            sample_x = 0
            n_burst = 1
            if sr_row != stim_record.shape[0] - 1:
                flipped = 1
                if directions[sample_y] == VAL_RIGHT:
                    flipped = -1
                mark_sound_location(stim_record,
                                    target,
                                    sr_row,
                                    flipped)
    return stim_record


def make_bin_stim_record(
        empty_record: list,
        directions: list,
        conditions: list) -> list:
    """Fills in stim_record based on stimulus in matfile and parameters in
    settings. Returns complete record."""
    # set tracking variables
    stim_record = empty_record.copy()
    sample_x = 0
    sample_y = 0
    times_rep = 0
    n_burst = 0
    max_reps = S_LEN + 1
    # iterate through rows of empty record and fill in values
    for sr_row in range(stim_record.shape[0]):
        # get directions
        flipped = 1
        if directions[sample_y] == VAL_RIGHT:
            flipped = -1
        # repeat until reaching max_reps and then don't change values
        if times_rep == max_reps:
            times_rep += 1
        # insert location of audio
        elif times_rep < max_reps:
            # if still on sound
            if sample_x < S_LEN:
                # get location
                target = conditions[sample_y, sample_x]
                # add if bursts less than N_BURSTS
                if n_burst < N_BURSTS:
                    mark_bin_location(stim_record, target, sr_row, flipped)
                    n_burst += 1
                # if bursts complete, increase x_sample and reset bursts
                else:
                    sample_x += 1
                    n_burst = 0
                # mark first location of next burst if not the last sound
                    if sample_x < 9 and times_rep != max_reps:
                        n_burst = 1
                        mark_bin_location(
                            stim_record,
                            target,
                            sr_row,
                            flipped)
                    elif times_rep != max_reps - 1:
                        n_burst = 1
                        target = conditions[sample_y, 0]
                        mark_bin_location(
                            stim_record,
                            target,
                            sr_row,
                            flipped)
                    else:
                        times_rep += 1
            # increase repreat counter and reset sample x index
            else:
                times_rep += 1
                sample_x = 0
                if times_rep != max_reps:
                    n_burst += 1
                    mark_bin_location(
                        stim_record,
                        target,
                        sr_row,
                        flipped)
                else:
                    n_burst = 0
        # plot blank space until reaching trial time
        elif (sr_row % 1600) != 1599:
            times_rep += 1
        # move on to next block of sounds
        else:
            sample_y += 1
            times_rep = 0
            sample_x = 0
            n_burst = 1
            if sr_row != stim_record.shape[0] - 1:
                flipped = 1
                if directions[sample_y] == VAL_RIGHT:
                    flipped = -1
                mark_bin_location(
                    stim_record,
                    target,
                    sr_row,
                    flipped)
    return stim_record


def stimulus_creation(path: str,
                      stim_space: tuple,
                      out_path: str = None,
                      save_fig: bool = True,
                      save_bin: bool = True,
                      save_val: bool = True) -> list:
    """Creates Python compatable stimulus design from .mat files.
    Takes path to subject data and stimulus space representation as a
    parameter. Path should contain .mat files generated from trials containing
    stimulus information. Stimulus design can be saved as a figure, a file
    containing the stimulus represented in binary, and/or a file containing the
    stimulus in its original values. Returns stimulus created."""

    # get matfiles
    matfiles = get_matfiles(path)
    if len(matfiles) == 0:
        print("No matfiles found. Please check path and try again")
        return matfiles

    # start iterating through matfiles and creating stim records
    for matfile_idx in len(range(matfiles)):
        # set matfile
        matfile = matfiles[matfile_idx]
        # get conditions array
        conditions = get_conditions(matfile)
        # check that conditions were found
        if len(conditions == 0):
            print("Conditions not found for this matfile.")
            continue
        # change conditions to numpy array
        conditions = np.array(conditions)
        # separate directions from actual conditions
        directions, conditions = get_directions(conditions)
        if len(directions) == 0 or len(conditions == 0):
            print("Getting directions/conditions raised an index error.")
            print("Check [DIR_COL] and [SEL_COL].")
            continue
        # get empty stim_record
        stim_record = make_empty_record()
        # fill in stim_record
        val_stim_record = make_val_stim_record(stim_record)
        bin_stim_record = make_bin_stim_record(stim_record)
        # save stim records
        if save_fig:
            save_record(
                val_stim_record,
                matfile_idx,
                out_path,
                'fig')
        if save_bin:
            save_record(
                bin_stim_record,
                matfile_idx,
                out_path,
                'bin'
            )
        if save_val:
            save_record(
                val_stim_record,
                matfile_idx,
                out_path,
                'bin'
            )
