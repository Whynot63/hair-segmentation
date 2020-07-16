import os
import shutil
import glob
import numpy as np

from tqdm import trange

from utils import DATASET_OUTPUT_DIR, create_dataset_dirs


DATASET_PATH = "Figaro1k/"


def get_images_paths(split):
    assert split in ["Training", "Testing"]
    return glob.glob(os.path.join(DATASET_PATH, "Original", split, "*"))


def get_image_mask_pair(image_path):
    return (
        image_path,
        image_path.replace("Original", "GT").replace("-org.jpg", "-gt.pbm"),
    )


if __name__ == "__main__":
    train_images = get_images_paths("Training")
    test_images = get_images_paths("Testing")

    train = [get_image_mask_pair(train_image) for train_image in train_images]
    test = [get_image_mask_pair(test_image) for test_image in test_images]

    create_dataset_dirs()

    for split, split_data in [("train", train), ("test", test)]:
        for i in trange(len(split_data), desc=split):
            image, mask = split_data[i]
            shutil.copyfile(
                image,
                os.path.join(
                    DATASET_OUTPUT_DIR, split, "image", f"Figaro1K{i}.jpg"
                ),
            )
            shutil.copyfile(
                mask,
                os.path.join(
                    DATASET_OUTPUT_DIR, split, "mask", f"Figaro1K{i}.png"
                ),
            )

