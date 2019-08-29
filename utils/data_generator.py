import glob
import os
import sys

import cv2
from keras.preprocessing.image import load_img
from keras.utils import Sequence
import numpy as np

from . import config


class DataGenerator(Sequence):
    """Corresponding filenames in source and target directories must be identical."""

    def __init__(self, source_dir, target_dir, batch_size, shuffle=True):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_filenames = self._get_img_filenames(source_dir)
        if self.shuffle:
            np.random.shuffle(self.img_filenames)

    def __getitem__(self, batch_num):
        print(batch_num)
        n_imgs = len(self.img_filenames)
        idx_start = batch_num * self.batch_size
        idx_end = min((batch_num+1) * self.batch_size, n_imgs)
        img_filenames_batch = self.img_filenames[idx_start:idx_end]
        imgs_source, imgs_target = self._get_batch(img_filenames_batch)

        return imgs_source, imgs_target

    def __len__(self):
        return int(np.ceil(len(self.img_filenames) / self.batch_size))

    def get_labels_fake(self):
        return np.zeros((self.batch_size, config.IMG_PATCH_HEIGHT, config.IMG_PATCH_WIDTH, 1))

    def get_labels_real(self):
        return np.ones((self.batch_size, config.IMG_PATCH_HEIGHT, config.IMG_PATCH_WIDTH, 1))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_filenames)

    def _get_batch(self, img_filenames_batch):
        batch_shape = (self.batch_size,) + config.IMG_SHAPE
        imgs_source = np.empty(batch_shape)
        imgs_target = np.empty(batch_shape)

        for idx, img_filename in enumerate(img_filenames_batch):
            imgs_source[idx] = self._get_img(os.path.join(self.source_dir, img_filename))
            imgs_target[idx] = self._get_img(os.path.join(self.target_dir, img_filename))

        return imgs_source, imgs_target

    def _get_img(self, img_filepath):
        img = load_img(img_filepath)
        img = np.array(img)
        img = cv2.resize(img, (config.IMG_HEIGHT, config.IMG_WIDTH))
        # scale from [0,255] to [-1,1]
        img = (img - 127.5) / 127.5

        return img

    def _get_img_filenames(self, directory):
        return [os.path.basename(fp) for fp in glob.glob(f'{directory}/*.jpg')]
