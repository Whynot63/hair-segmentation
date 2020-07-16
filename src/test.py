import glob
import os
import time
import argparse

import cv2
import tensorflow as tf
import numpy as np


CKPT_PATH = "checkpoints/best.hdf5"


def get_args():
    parser = argparse.ArgumentParser(description="Hair Segmentation Demo")
    parser.add_argument(
        "--selfie_dir",
        help='Directory with selfie images to recognize. Takes only files with "%04d.jpg" format.',
        required=True,
    )
    parser.add_argument(
        "--output_dir", help="Directory with result masks.", required=True,
    )
    args = parser.parse_args()

    return args


def predict(image, height=224, width=224):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    pred = model.predict(im)
    mask = pred.reshape((224, 224))

    return mask


if __name__ == "__main__":
    args = get_args()

    model = tf.keras.models.load_model(CKPT_PATH)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for image_path in glob.glob(
        os.path.join(args.selfie_dir, "[0-9][0-9][0-9][0-9].jpg")
    ):
        img = cv2.imread(image_path)
        mask = predict(img)
        cv2.imwrite(
            os.path.join(args.output_dir, os.path.basename(image_path)),
            mask * 255,
        )

