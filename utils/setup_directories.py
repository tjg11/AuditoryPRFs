import os
import json
from dotenv import load_dotenv


def create_sub_folders():
    """Creates subdirectories in folder located at environment variable
    DATA_PATH for each subject id in environment variable SUB_IDS. Returns
    number of folders created."""

    # load environment variables
    load_dotenv()

    # set paths and subject ids
    data_path = os.getenv("DATA_PATH")
    sub_ids = json.loads(os.getenv("SUB_IDS"))

    # create subdirectory for each subject id\
    dir_count = 0
    for sub_id in sub_ids:
        new_path = os.path.join(data_path, sub_id)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            dir_count += 1

    # return number of created directories
    return dir_count


if __name__ == '__main__':
    print(f" Subject folders created: {create_sub_folders()}.")
