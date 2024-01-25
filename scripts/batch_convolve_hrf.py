from convolve_hrf import convolve_hrf
import os
import json
import pickle
import glob
from dotenv import load_dotenv


def batch_convolve(path, save_path):
    """
    Uses convolve_hrf from convolve_hrf.py to iterate through multiple stimulus
    pickle files. Takes the path to the target file and a path to save the
    convolved stimulus to as parameters. Returns None.
    """
    # open .pickle file
    with open(path, "rb") as f:
        record = pickle.load(f)
    # convolve loaded file
    convolved_record = convolve_hrf(record)
    # save convolved file
    with open(save_path, "wb") as f:
        pickle.dump(convolved_record, f)
    return


if __name__ == '__main__':
    # load environment variables
    load_dotenv()
    # get subject IDS and set paths
    subjects = json.loads(os.getenv("SUB_IDS"))
    paths_data = os.getenv("DATA_PATH")
    # iterate through subjects
    for subject in subjects:
        # set subject paths
        record_path = os.path.join(
            paths_data,
            subject,
            "stim_matricies",
            "*"
        )
        main_path = os.path.join(
            paths_data,
            subject,
            "convolved_matricies"
        )
        # create saving directory if it doesn't exist
        if not os.path.exists(main_path):
            os.makedirs(main_path)
        # get all stimulus files
        records = glob.glob(record_path)
        # convolve each file
        for idx in range(len(records)):
            # set individual save path
            save_path = os.path.join(
                main_path,
                f"convolved_stim{idx}.pickle"
            )
            batch_convolve(records[idx], save_path)
