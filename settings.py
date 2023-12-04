import numpy as np

# Settings for stimulus_design

# length of stimulus record
S_LEN = 60
# length of one trials (in ms)
TRIAL_LEN = 26 * 16000
# resolution of output (in ms)
RESOLUTION = 10  # ms
# index location of directions of auditory stimulus
DIR_LOC = 2
# distance markers for auditory stimulus
DIST = np.linspace(0, 60, 9)
# index locations of columns in condition where information is located
SEL_COL = [3, 4, 5, 6, 7, 8, 9, 10, 11]
# value representing left direction
VAL_LEFT = 1
# value representing right direction
VAL_RIGHT = 2
# value representing number of 'bursts' of single sound
N_BURSTS = 9
