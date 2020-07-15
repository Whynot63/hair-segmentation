import argparse

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
import keras

from keras.optimizers import Adadelta
import tensorflow as tf

from data_generator import dataGenerator
from HairSegNet import get_model
import datetime

import math
import os


def get_args():
    parser = argparse.ArgumentParser(description="Hair Segmentation")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--use_pretrained", type=bool, default=False)
    parser.add_argument("--path_model", default="checkpoints")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    return args


def setDevice(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"


def pathFolderCheckpoint(path_model):
    current = datetime.datetime.now()

    path_model = f'{path_model}/{current.strftime("%m-%d-%Y_%H-%M-%S")}'

    if not os.path.exists(path_model):
        os.makedirs(path_model)

    return path_model


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(
        drop, math.floor((1 + epoch) / epochs_drop)
    )
    return lrate


def getData(args):
    # Augmentation Data
    data_gen_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_data = dataGenerator(
        args.batch_size,
        data_gen_args,
        f"{args.data_dir}/train",
        "image",
        "mask",
        save_to_dir=None,
    )

    val_data = dataGenerator(
        args.batch_size,
        data_gen_args,
        f"{args.data_dir}/test",
        "image",
        "mask",
        save_to_dir=None,
    )

    return train_data, val_data


def getModel(args):
    model = get_model()

    model.compile(
        optimizer=Adadelta(learning_rate=1, rho=0.95, epsilon=1e-07),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def getCallback(args):

    model_checkpoint = ModelCheckpoint(
        os.path.join(
            args.path_model, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        ),
        monitor="loss",
        verbose=1,
        save_best_only=True,
    )
    csv_logger = CSVLogger(
        os.path.join(args.path_model, "model_history_log.csv"), append=True
    )

    lrate = LearningRateScheduler(step_decay)

    return [model_checkpoint, csv_logger]


if __name__ == "__main__":
    args = get_args()
    print(args)

    setDevice(args)

    size_data = len(os.listdir(f"{args.data_dir}/train/image"))
    args.steps_per_epoch = size_data // args.batch_size

    args.path_model = pathFolderCheckpoint(args.path_model)

    train_data, val_data = getData(args)

    model = getModel(args)

    callbacks = getCallback(args)
    model.fit_generator(
        train_data,
        callbacks=callbacks,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_data,
        validation_steps=50,
    )

