### Author: Taylor Garrison
### Description: Designed to take in a matlab stimulus for auditory trials and convert them
### to either image or binary files for use in following scripts. Stimulus space for this 
### experiment in audio (left -> right) represented in a numerical array starting at -30 and
### ending at 30 with 9 steps between the two. Stimuli can either travel from left to right or right
### to left, and fall into 3 categories: sequential, onset/offset, and random.

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os.path as op
import scipy.io, re, pickle, time, glob, os

# in_path for my local machine: ../../AMPB/data/*
# currently saves out into the same directory the script is located in
# Expected structure for in path: /subject_directories/ses-02/func/*.mat

def stimulus_creation(in_path, save_type):
    """Creates and saves out stimuli in binary or img format. Path to data to be transformed and saved
    entered in in_path. Type of file saved out defined by 
    save_type (1=binary, 2=binary[with actual distances], 3=image)"""
    # Input and output paths
    data_path = op.abspath(in_path)
    print(data_path)

    # Find all subjects
    sub_ids = glob.glob(data_path)
    sub_ids = sub_ids[7:len(sub_ids)]
    print(f"All sub ids: \n {sub_ids}.\n")

    # Add on wildcard path to each subject id
    for sub_idx in range(len(sub_ids)):
        sub_ids[sub_idx] = op.join(sub_ids[sub_idx], 'ses-02', 'func', '*.mat')

    # Populate list with path to each matfile
    all_matfiles = []
    for sub_id in sub_ids:
        matfiles = glob.glob(sub_id)
        all_matfiles += matfiles

    print(f"Number of .mat files: {len(all_matfiles)}.")

    # Create stim space
    space = np.linspace(-30, 30, 9)
    print(f"Stimulus space: {space}") # remeber 0-based index

    # Show mat file info for one subject
    test_sub = all_matfiles[2]
    mat_file = scipy.io.loadmat(test_sub)
    conditions = mat_file['conditions'][0][0][0]

    # get condition variables
    cond_as_numpy = np.matrix(np.array(conditions))
    cond_as_numpy = cond_as_numpy[:, [1,2]]
    print(f"Conditions for random mat file (category, direction): {cond_as_numpy}")

    # Create directories for each subject

    for mat_idx in range(len(all_matfiles)):
        fname = all_matfiles[mat_idx]
        bname = op.basename(fname)
        subject_name = re.sub("^(sub-[\w\d]+)_.+", "\\1", bname)
        if not op.exists(subject_name):
            os.makedirs(subject_name)
    if save_type == 1:
        # Get pickle file for each matfile (BINARY)
        SOUND_LEN = 90   # ms
        SAMPLE_LEN = 9
        SAMPLE_REP = 10
        N_TRIALS  = 26
        TRIAL_LEN = 26 * 16000 # ms
        RESOLUTION = 10 # ms
        DIR_LOC = 2 # index in conditions
        DIST = np.linspace(-30, 30, 9)

        tic = time.perf_counter()

        # process each matfile
        for mat_idx in range(len(all_matfiles)):
            
            # load matfile conditions
            SEL_COL = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            mat_file = scipy.io.loadmat(all_matfiles[mat_idx])
            conditions = mat_file['conditions'][0][0][0]

            # get condition variables
            cond_as_numpy = np.matrix(np.array(conditions))
            directions = cond_as_numpy[:, DIR_LOC]
            cond_as_numpy = cond_as_numpy[:, SEL_COL]
            

            rest_time = TRIAL_LEN - (SOUND_LEN * SAMPLE_LEN * SAMPLE_REP * N_TRIALS)
            
            # create stim record matrix
            n_bins = TRIAL_LEN / RESOLUTION
            stim_record = np.zeros((int(n_bins), SAMPLE_LEN))
            
            # loop variables
            sample_idx_x = 0
            sample_idx_y = 0
            times_rep = 0
            n_burst = 0
            max_reps = 10
            FLIP = 1
            
            for row_in_stim_record in range(stim_record.shape[0]):
                # plot the repeated sound burst 10 times (one block)
                
                # determine direction
                if directions[sample_idx_y] == 2:
                    FLIP = -1
                else:
                    FLIP = 1
                
                # Repeat until reaching 10 reps and then keep zeros
                if times_rep == max_reps:
                    times_rep += 1
                
                # insert location of sound
                elif times_rep < max_reps:
                    if sample_idx_x < 9:
                        # plot each sound (9 sounds)
                        target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                        distance = DIST[target_idx]
                        if FLIP == -1:
                            target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                        if n_burst <= 8:
                            
                            stim_record[row_in_stim_record, target_idx] = 1
                            n_burst += 1
                        else:
                            sample_idx_x += 1
                            n_burst = 0
                            if sample_idx_x < 9 and times_rep != max_reps:
                                n_burst = 1
                                target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                                distance = DIST[target_idx]
                                if FLIP == -1:
                                    target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                                stim_record[row_in_stim_record, target_idx] = 1
                            elif times_rep != max_reps - 1:
                                n_burst = 1
                                target_idx = int(cond_as_numpy[sample_idx_y, 0] - 1)
                                distance = DIST[target_idx]
                                if FLIP == -1:
                                    target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                                stim_record[row_in_stim_record, target_idx] = 1
                            else:
                                times_rep += 1
                    # increase repeat counter and reset sample x index
                    else:
                        times_rep += 1
                        sample_idx_x = 0
                        if times_rep != max_reps:
                            n_burst += 1
                            target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                            distance = DIST[target_idx]
                            if FLIP == -1:
                                target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                            stim_record[row_in_stim_record, target_idx] = 1
                        else:
                            n_burst = 0
                # plot blank space until reaching 16 sec
                elif (row_in_stim_record % 1600) != 1599:
                    times_rep += 1
                # move on to next block
                else:
                    sample_idx_y += 1
                    times_rep = 0
                    sample_idx_x = 0
                    n_burst = 1
                    if row_in_stim_record != stim_record.shape[0] - 1:
                        target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                        if directions[sample_idx_y] == 2:
                            FLIP = -1
                        else:
                            FLIP = 1
                        distance = DIST[target_idx]
                        if FLIP == -1:
                            target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                        stim_record[row_in_stim_record, target_idx] = 1
            # save matrix as image
            fname = all_matfiles[mat_idx]
            bname = op.basename(fname)
            subject_name = re.sub("^(sub-[\w\d]+)_.+", "\\1", bname)
            run_index = re.sub(".+_run-(\d+)_.+", "\\1", bname)
            img_name = 'binary_' + subject_name + '_' + run_index + ".pickle"
            img_name = op.join(subject_name, img_name)
            with open(img_name, "wb") as f:
                pickle.dump(stim_record, f) # variable you want to save first then file
        toc = time.perf_counter()
        return(f"Finished saving pickle files. Time taken: {toc - tic}.")
    elif save_type == 2:
        # Get the pickle file for each mat file in matfiles (DISTANCE)
        SOUND_LEN = 90   # ms
        SAMPLE_LEN = 9
        SAMPLE_REP = 10
        N_TRIALS  = 26
        TRIAL_LEN = 26 * 16000 # ms
        RESOLUTION = 10 # ms
        DIR_LOC = 2 # index in conditions
        DIST = np.linspace(-30, 30, 9)

        tic = time.perf_counter()

        # process each matfile
        for mat_idx in range(len(all_matfiles)):
            
            # load matfile conditions
            SEL_COL = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            mat_file = scipy.io.loadmat(all_matfiles[mat_idx])
            conditions = mat_file['conditions'][0][0][0]

            # get condition variables
            cond_as_numpy = np.matrix(np.array(conditions))
            directions = cond_as_numpy[:, DIR_LOC]
            cond_as_numpy = cond_as_numpy[:, SEL_COL]
            

            rest_time = TRIAL_LEN - (SOUND_LEN * SAMPLE_LEN * SAMPLE_REP * N_TRIALS)
            
            # create stim record matrix
            n_bins = TRIAL_LEN / RESOLUTION
            stim_record = np.zeros((int(n_bins), SAMPLE_LEN))
            
            # loop variables
            sample_idx_x = 0
            sample_idx_y = 0
            times_rep = 0
            n_burst = 0
            max_reps = 10
            FLIP = 1
            
        for mat_idx in range(len(all_matfiles)):
            
            # load matfile conditions
            SEL_COL = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            mat_file = scipy.io.loadmat(all_matfiles[mat_idx])
            conditions = mat_file['conditions'][0][0][0]

            # get condition variables
            cond_as_numpy = np.matrix(np.array(conditions))
            directions = cond_as_numpy[:, DIR_LOC]
            cond_as_numpy = cond_as_numpy[:, SEL_COL]
            

            rest_time = TRIAL_LEN - (SOUND_LEN * SAMPLE_LEN * SAMPLE_REP * N_TRIALS)
            
            # create stim record matrix
            n_bins = TRIAL_LEN / RESOLUTION
            stim_record = np.zeros((int(n_bins), SAMPLE_LEN))
            
            # loop variables
            sample_idx_x = 0
            sample_idx_y = 0
            times_rep = 0
            n_burst = 0
            max_reps = 10
            FLIP = 1
            
            for row_in_stim_record in range(stim_record.shape[0]):
                # plot the repeated sound burst 10 times (one block)
                if directions[sample_idx_y] == 2:
                    FLIP = -1
                else:
                    FLIP = 1
                if times_rep == max_reps:
                    times_rep += 1
                elif times_rep < max_reps:
                    if sample_idx_x < 9:
                        # plot each sound (9 sounds)
                        target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                        distance = DIST[target_idx]
                        if FLIP == -1:
                            target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                        if n_burst <= 8:
                            
                            stim_record[row_in_stim_record, target_idx] = distance * FLIP
                            n_burst += 1
                        else:
                            sample_idx_x += 1
                            n_burst = 0
                            if sample_idx_x < 9 and times_rep != max_reps:
                                n_burst = 1
                                target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                                distance = DIST[target_idx]
                                if FLIP == -1:
                                    target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                                stim_record[row_in_stim_record, target_idx] = distance * FLIP
                            elif times_rep != max_reps - 1:
                                n_burst = 1
                                target_idx = int(cond_as_numpy[sample_idx_y, 0] - 1)
                                distance = DIST[target_idx]
                                if FLIP == -1:
                                    target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                                stim_record[row_in_stim_record, target_idx] = distance * FLIP
                            else:
                                times_rep += 1
                    # increase repeat counter and reset sample x index
                    else:
                        times_rep += 1
                        sample_idx_x = 0
                        if times_rep != max_reps:
                            n_burst += 1
                            target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                            distance = DIST[target_idx]
                            if FLIP == -1:
                                target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                            stim_record[row_in_stim_record, target_idx] = distance * FLIP
                        else:
                            n_burst = 0
                # plot blank space until reaching 16 sec
                elif (row_in_stim_record % 1600) != 1599:
                    times_rep += 1
                # move on to next block
                else:
                    sample_idx_y += 1
                    times_rep = 0
                    sample_idx_x = 0
                    n_burst = 1
                    if row_in_stim_record != stim_record.shape[0] - 1:
                        target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                        if directions[sample_idx_y] == 2:
                            FLIP = -1
                        else:
                            FLIP = 1
                        distance = DIST[target_idx]
                        if FLIP == -1:
                            target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                        stim_record[row_in_stim_record, target_idx] = distance * FLIP
            # save matrix as image
            fname = all_matfiles[mat_idx]
            bname = op.basename(fname)
            subject_name = re.sub("^(sub-[\w\d]+)_.+", "\\1", bname)
            run_index = re.sub(".+_run-(\d+)_.+", "\\1", bname)
            img_name = subject_name + '_' + run_index + ".pickle"
            img_name = op.join(subject_name, img_name)
            with open(img_name, "wb") as f:
                pickle.dump(stim_record, f) # variable you want to save first then file
        toc = time.perf_counter()
        return(f"Finished saving pickle files. Time taken: {toc - tic}.")

    elif save_type == 3:
        # Get the image for each mat file in matfiles
        SOUND_LEN = 90   # ms
        SAMPLE_LEN = 9
        SAMPLE_REP = 10
        N_TRIALS  = 26
        TRIAL_LEN = 26 * 16000 # ms
        RESOLUTION = 10 # ms
        DIR_LOC = 2 # index in conditions
        DIST = np.linspace(-30, 30, 9)

        tic = time.perf_counter()

        # process each matfile
        for mat_idx in range(len(all_matfiles)):
            
            # load matfile conditions
            SEL_COL = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            mat_file = scipy.io.loadmat(all_matfiles[mat_idx])
            conditions = mat_file['conditions'][0][0][0]

            # get condition variables
            cond_as_numpy = np.matrix(np.array(conditions))
            directions = cond_as_numpy[:, DIR_LOC]
            cond_as_numpy = cond_as_numpy[:, SEL_COL]
            

            rest_time = TRIAL_LEN - (SOUND_LEN * SAMPLE_LEN * SAMPLE_REP * N_TRIALS)
            
            # create stim record matrix
            n_bins = TRIAL_LEN / RESOLUTION
            stim_record = np.zeros((int(n_bins), SAMPLE_LEN))
            
            # loop variables
            sample_idx_x = 0
            sample_idx_y = 0
            times_rep = 0
            n_burst = 0
            max_reps = 10
            FLIP = 1
            
            for row_in_stim_record in range(stim_record.shape[0]):
                # plot the repeated sound burst 10 times (one block)
                if directions[sample_idx_y] == 2:
                    FLIP = -1
                else:
                    FLIP = 1
                if times_rep == max_reps:
                    times_rep += 1
                elif times_rep < max_reps:
                    if sample_idx_x < 9:
                        # plot each sound (9 sounds)
                        target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                        distance = DIST[target_idx]
                        if FLIP == -1:
                            target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                        if n_burst <= 8:
                            
                            stim_record[row_in_stim_record, target_idx] = distance * FLIP
                            n_burst += 1
                        else:
                            sample_idx_x += 1
                            n_burst = 0
                            if sample_idx_x < 9 and times_rep != max_reps:
                                n_burst = 1
                                target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                                distance = DIST[target_idx]
                                if FLIP == -1:
                                    target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                                stim_record[row_in_stim_record, target_idx] = distance * FLIP
                            elif times_rep != max_reps - 1:
                                n_burst = 1
                                target_idx = int(cond_as_numpy[sample_idx_y, 0] - 1)
                                distance = DIST[target_idx]
                                if FLIP == -1:
                                    target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                                stim_record[row_in_stim_record, target_idx] = distance * FLIP
                            else:
                                times_rep += 1
                    # increase repeat counter and reset sample x index
                    else:
                        times_rep += 1
                        sample_idx_x = 0
                        if times_rep != max_reps:
                            n_burst += 1
                            target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                            distance = DIST[target_idx]
                            if FLIP == -1:
                                target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                            stim_record[row_in_stim_record, target_idx] = distance * FLIP
                        else:
                            n_burst = 0
                # plot blank space until reaching 16 sec
                elif (row_in_stim_record % 1600) != 1599:
                    times_rep += 1
                # move on to next block
                else:
                    sample_idx_y += 1
                    times_rep = 0
                    sample_idx_x = 0
                    n_burst = 1
                    if row_in_stim_record != stim_record.shape[0] - 1:
                        target_idx = int(cond_as_numpy[sample_idx_y, sample_idx_x] - 1)
                        if directions[sample_idx_y] == 2:
                            FLIP = -1
                        else:
                            FLIP = 1
                        distance = DIST[target_idx]
                        if FLIP == -1:
                            target_idx = (abs(target_idx - SAMPLE_LEN)) - 1
                        stim_record[row_in_stim_record, target_idx] = distance * FLIP
            # save matrix as image
            fname = all_matfiles[mat_idx]
            bname = op.basename(fname)
            subject_name = re.sub("^(sub-[\w\d]+)_.+", "\\1", bname)
            run_index = re.sub(".+_run-(\d+)_.+", "\\1", bname)
            img_name = subject_name + '_' + run_index + ".png"
            img_name = op.join(subject_name, img_name)
            plt.imsave(img_name, stim_record)
        toc = time.perf_counter()

        return(f"Finished saving images. Time taken: {toc - tic}.")
    else:
        return("Invalid save_type entered. Please entered 1, 2, or 3.")
    

# Run script on local machine
if __name__ == "__main__":
    print(stimulus_creation("OneDrive - UW/AMPB/data/*", 1))