import os
import json
from dotenv import load_dotenv
from stimulus_design import stimulus_design


def batch_stimulus_design():
    """Takes no parameters. Utilizes paths stored in .env to create stimulus
    files for each subject listed in SUBJECT_IDS. Returns number of files
    created."""

    # set count for files created
    file_count = 0

    # load environment variables
    load_dotenv()

    # set base path
    base_path = os.getenv("ORIG_PATH")

    # load subject ids
    sub_ids = json.loads(os.getenv("SUB_IDS"))

    # set subject data path (out path)
    save_path = os.getenv("DATA_PATH")

    # iterate through subject ids
    for sub_id in sub_ids:
        # redirect path to where matfiles are
        mat_path = os.path.join(
            base_path,
            sub_id,
            'ses-02',
            'func'
        )

        # set subject specific save path
        out_path = os.path.join(
            save_path,
            sub_id,
            "stim_matricies"
        )
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # run stimulus_design
        files_created = stimulus_design(mat_path,
                                        out_path=out_path,
                                        save_bin=True)
        file_count += files_created

    return file_count


if __name__ == '__main__':
    file_count = batch_stimulus_design()
    print(f"Stimulus files created: {file_count}")
