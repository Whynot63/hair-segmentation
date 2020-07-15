import os
import shutil
import glob
import numpy as np

from tqdm import trange

from utils import DATASET_OUTPUT_DIR, create_dataset_dirs


DATASET_PATH = "CelebAMask-HQ/"
TRAIN_TEST_SPLIT_RATIO = 0.9


def get_masks_paths():
    return glob.glob(
        os.path.join(DATASET_PATH, "CelebAMask-HQ-mask-anno/*/*_hair.png")
    )


def get_image_mask_pair(mask_path):
    mask_id = int(mask_path.split("/")[-1].replace("_hair.png", ""))
    image_path = os.path.join("CelebAMask-HQ/CelebA-HQ-img", f"{mask_id}.jpg")
    return (image_path, mask_path)


if __name__ == "__main__":
    masks = get_masks_paths()
    images_with_mask = list(map(get_image_mask_pair, masks))
    n_samples = len(images_with_mask)

    idx = np.random.permutation(np.arange(n_samples))
    train = np.take(
        images_with_mask,
        idx[: int(n_samples * TRAIN_TEST_SPLIT_RATIO)],
        axis=0,
    )
    test = np.take(
        images_with_mask,
        idx[int(n_samples * TRAIN_TEST_SPLIT_RATIO) :],
        axis=0,
    )

    create_dataset_dirs()

    for split, split_data in [("train", train), ("test", test)]:
        for i in trange(len(split_data), desc=split):
            image, mask = split_data[i]
            shutil.copyfile(
                image,
                os.path.join(
                    DATASET_OUTPUT_DIR, split, "image", f"CelebA{i}.jpg"
                ),
            )
            shutil.copyfile(
                mask,
                os.path.join(
                    DATASET_OUTPUT_DIR, split, "mask", f"CelebA{i}.png"
                ),
            )

