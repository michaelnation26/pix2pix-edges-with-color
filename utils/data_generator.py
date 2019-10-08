import glob
import os
import sys

import cv2
from keras.preprocessing.image import load_img
from keras.utils import Sequence
import numpy as np

from . import config


def get_img_for_model(img_filepath):
    """Loads image from disk.
    Resizes image to height and width specified in config file.
    Rescales image from [0, 255] to [-1, 1]
    """
    img = load_img(img_filepath, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
    img = np.array(img)
    img = (img - 127.5) / 127.5

    return img

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
        imgs_source, imgs_target, discriminator_labels_real, discriminator_labels_fake \
            = self._get_batch(img_filenames_batch)

        return imgs_source, imgs_target, discriminator_labels_real, discriminator_labels_fake

    def __len__(self):
        return int(np.ceil(len(self.img_filenames) / self.batch_size))

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.img_filenames)

    def _draw_color_circles_on_src_img(self, img_src, img_target):
        non_white_coords = self._get_non_white_coordinates(img_target)
        for center_y, center_x in non_white_coords:
            self._draw_color_circle_on_src_img(img_src, img_target, center_y, center_x)

    def _draw_color_circle_on_src_img(self, img_src, img_target, center_y, center_x):
        assert(img_src.shape == img_target, "Image source and target must have same shape.")

        y0, y1, x0, x1 = self._get_color_point_bbox_coords(center_y, center_x)
        color = np.mean(img_target[y0:y1, x0:x1], axis=(0, 1))
        img_src[y0:y1, x0:x1] = color

    def _get_batch(self, img_filenames_batch):
        batch_size = len(img_filenames_batch)
        batch_shape = (batch_size,) + config.IMG_SHAPE
        img_sources = np.empty(batch_shape)
        img_targets = np.empty(batch_shape)

        for idx, img_filename in enumerate(img_filenames_batch):
            img_source, img_target = self._get_img_source_and_img_target(img_filename)
            img_sources[idx] = img_source
            img_targets[idx] = img_target

        discriminator_labels_real = self._get_discriminator_labels_real(batch_size)
        discriminator_labels_fake = self._get_discriminator_labels_fake(batch_size)

        return img_sources, img_targets, discriminator_labels_real, discriminator_labels_fake

    def _get_color_point_bbox_coords(self, center_y, center_x):
        radius = config.USER_COLOR_POINTS_RADIUS
        y0 = max(0, center_y-radius+1)
        y1 = min(config.IMG_HEIGHT, center_y+radius)
        x0 = max(0, center_x-radius+1)
        x1 = min(config.IMG_WIDTH, center_x+radius)

        return y0, y1, x0, x1

    def _get_discriminator_labels_fake(self, batch_size):
        return np.zeros((batch_size, config.IMG_PATCH_HEIGHT, config.IMG_PATCH_WIDTH, 1))

    def _get_discriminator_labels_real(self, batch_size):
        return np.ones((batch_size, config.IMG_PATCH_HEIGHT, config.IMG_PATCH_WIDTH, 1))

    def _get_img_filenames(self, directory):
        return [os.path.basename(fp) for fp in glob.glob(f'{directory}/*.jpg')]

    def _get_img_source_and_img_target(self, img_filename):
        img_source = get_img_for_model(os.path.join(self.source_dir, img_filename))
        img_target = get_img_for_model(os.path.join(self.target_dir, img_filename))

        if self.is_training:
            self._draw_color_circles_on_src_img(img_source, img_target)
            # data augmentation
            if np.random.random_sample() > 0.5:
                img_source = np.fliplr(img_source)
                img_target = np.fliplr(img_target)

        return img_source, img_target

    def _get_non_white_coordinates(self, img):
        non_white_mask = np.sum(img, axis=-1) < 2.75
        non_white_y, non_white_x = np.nonzero(non_white_mask)

        # randomly sample non-white coordinates
        n_non_white = len(non_white_y)
        n_color_points = min(n_non_white, config.USER_COLOR_POINTS_PER_IMG)
        idxs = np.random.choice(n_non_white, n_color_points, replace=False)
        non_white_coords = zip(non_white_y[idxs], non_white_x[idxs])

        return non_white_coords
