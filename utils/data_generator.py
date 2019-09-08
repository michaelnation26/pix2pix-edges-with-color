import glob
import os
import sys

import cv2
from keras.preprocessing.image import load_img
from keras.utils import Sequence
import numpy as np

from . import config


class DataGenerator(Sequence):
    """Filenames for source and target image pairs must be identical.
    e.g. data/training/source/my_img.jpg -> data/training/target/my_img.jpg
    """

    def __init__(self, source_dir, target_dir, batch_size, is_training):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.is_training = is_training

        self.img_filenames = self._get_img_filenames(source_dir)
        if self.is_training:
            np.random.shuffle(self.img_filenames)
        else:
            self.img_filenames = np.sort(self.img_filenames)

    def __getitem__(self, batch_num):
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
        if self.is_training:
            np.random.shuffle(self.img_filenames)

    def _draw_color_circles_on_src_img(self, img_src, img_target):
        non_white_coords = np.where(~np.all(img_target == 255, axis=2))
        idxs = np.random.choice(len(non_white_coords[0]), config.USER_COLOR_POINTS_PER_IMG, replace=False)
        for idx in idxs:
            self._draw_color_circle_on_src_img(
                img_src, img_target, center_y=non_white_coords[0][idx], center_x=non_white_coords[1][idx])

    def _draw_color_circle_on_src_img(self, img_src, img_target, center_y, center_x):
        assert(img_src.shape == img_target, "Image source and target must have same shape.")
        color = self._get_mean_color(img_target, center_y, center_x)
        cv2.circle(img_src, (center_x, center_y), config.USER_COLOR_POINTS_CIRCLE_RADIUS, color, cv2.FILLED)

    def _get_batch(self, img_filenames_batch):
        batch_shape = (self.batch_size,) + config.IMG_SHAPE
        img_sources = np.empty(batch_shape)
        img_targets = np.empty(batch_shape)

        for idx, img_filename in enumerate(img_filenames_batch):
            img_source, img_target = self._get_img_source_and_img_target(img_filename)
            img_sources[idx] = img_source
            img_targets[idx] = img_target

        return img_sources, img_targets

    def _get_img(self, img_filepath):
        img = load_img(img_filepath)
        img = np.array(img)
        # scale from [0,255] to [-1,1]
        img = (img - 127.5) / 127.5

        return img

    def _get_img_filenames(self, directory):
        return [os.path.basename(fp) for fp in glob.glob(f'{directory}/*.jpg')]

    def _get_img_source_and_img_target(self, img_filename):
        img_source = self._get_img(os.path.join(self.source_dir, img_filename))
        img_target = self._get_img(os.path.join(self.target_dir, img_filename))

        if self.is_training:
            self._draw_color_circles_on_src_img(img_source, img_target)
            # data augmentation
            if np.random.random_sample() > 0.5:
                img_source = np.fliplr(img_source)
                img_target = np.fliplr(img_target)

        return img_source, img_target

    def _get_mean_color(self, img, center_y, center_x):
        radius = config.USER_COLOR_POINTS_CIRCLE_RADIUS
        h, w = img.shape[:2]

        y0 = max(0, center_y-radius)
        y1 = min(h, center_y+radius)
        x0 = max(0, center_x-radius)
        x1 = min(w, center_x+radius)
        mean_color = np.mean(img[y0:y1, x0:x1], axis=(0, 1))

        return mean_color.tolist()
