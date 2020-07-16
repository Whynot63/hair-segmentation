import os

DATASET_OUTPUT_DIR = "./dataset"


def create_dataset_dirs():
    for split in ["train", "test"]:
        for set_ in ["image", "mask"]:
            dir_path = os.path.join(DATASET_OUTPUT_DIR, split, set_)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
