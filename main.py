from utils.convolve_hrf import convolve_hrf
from utils.get_htmls import get_htmls
from utils.get_rois import get_rois
import numpy as np
from os import getenv
from dotenv import load_dotenv


run_convolved = False
run_htmls = False
run_rois = True
get_stimuli = True
# Run through all sighted particpants and run numbers to get all PRFs
load_dotenv()
subject_ids = getenv('SUB_IDS')

run_numbers = ["01", "02", "03", "04", "05", "06"]

area_thresholds = [25, 50, 75, 100]
p_thresholds = [0.05, 0.01, 0.005, 0.001]
complete_thresholds = np.vstack(
    np.meshgrid(
        area_thresholds, p_thresholds)).reshape(2, -1).T
all_subruns = np.vstack(np.meshgrid(subject_ids, run_numbers)).reshape(2, -1).T

print(f"First 9 items in all_subruns:\n {all_subruns[0:9]}.")
print(f"Shape of all_subruns: {all_subruns.shape}.")
if run_convolved:

    count = 0
    for sub, run in all_subruns:
        print(f"""Convolving for: {sub, run}.
              Finished {count} / {all_subruns.shape[0]} tasks.\n """)
        convolve_hrf(sub, run)
        count += 1


# test_sub, test_run = all_subruns[0]
# print(test_sub, test_run)

# print(find_prf(test_sub, test_run, (20, 23), (20, 23), (20, 22), (800, 600)))
# where = os.path.abspath("OneDrive - UW/AMPB/data")
where = "../../AMPB/data"

if run_htmls:
    for sub in subject_ids:
        print(f"Getting contrast HTML viewer for {sub}.")
        get_htmls(sub, 3, where, save_bg_image=True)

if run_rois:
    for sub in subject_ids:
        for a_thresh, p_thresh in complete_thresholds:
            print(f"""Getting ROIS for {sub}
                  with {a_thresh} and {p_thresh}.""")
            get_rois(sub,
                     target_area=a_thresh,
                     p_threshold=p_thresh)
