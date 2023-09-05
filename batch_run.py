from convolve_hrf import convolve_hrf
from get_htmls import get_htmls
import numpy as np


run_convolved = False
run_htmls = True
# Run through all sighted particpants and run numbers to get all PRFs

subject_ids = ["sub-NSxGxBAx1970",
               "sub-NSxGxBYx1981",
               "sub-NSxGxHKx1965",
               "sub-NSxGxHNx1952",
               "sub-NSxGxIFx1991",
               "sub-NSxGxNXx1990",
               "sub-NSxGxRFx1978",
               "sub-NSxGxXJx1998",
               "sub-NSxGxYRx1992",
               "sub-NSxLxATx1954",
               "sub-NSxLxBNx1985",
               "sub-NSxLxIUx1994",
               "sub-NSxLxPQx1973",
               "sub-NSxLxQFx1997",
               "sub-NSxLxQUx1953",
               "sub-NSxLxVDx1987",
               "sub-NSxLxVJx1998",
               "sub-NSxLxYKx1964",
               "sub-NSxLxYNx1999"]

run_numbers = ["01", "02", "03", "04", "05", "06"]

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
    for sub, run in all_subruns:
        print(f"Getting contrast HTML viewer for {sub, run}.")
        get_htmls(sub, run[1], where, save_nifti=True)
