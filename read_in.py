
import os
import numpy as np
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
import hyperparameters as hp

class Datasets():
    def __init__(self, dat_dir):
        self.Path_A = os.path.join(dat_dir, 'A')
        self.Path_B = os.path.join(dat_dir, 'B')

        self.train_A = self.push_data(self.Path_A)
        self.train_B = self.push_data(self.Path_B)
        self.test_A = self.push_data(self.Path_A, False)
        self.test_B = self.push_data(self.Path_B, False)

    def preprocess_sequence(self, img, augment):
        # Required for some augmentations
        img = img.astype(np.uint8)
        if augment:
            augmentations = ia.augmenters.Sequential([
                iaa.Resize(int(1.1*hp.img_size)),
                iaa.Fliplr(0.5),
                iaa.Sometimes(0.4,
                iaa.Rotate((-30, 30))),
                iaa.Sometimes(0.4,
                iaa.Affine(scale=(0.9, 1.2))),
                iaa.Sometimes(0.5,
                iaa.PerspectiveTransform(scale=(0.01, 0.20))),
                # Crop/resize image to proper dimension
                iaa.CropToFixedSize(hp.img_size, hp.img_size), iaa.Resize(hp.img_size),
                iaa.Sometimes(0.3,
                iaa.SaltAndPepper(0.01)),
                iaa.CLAHE(to_colorspace='HSV')])
        else:
            augmentations = iaa.CLAHE(to_colorspace='HSV')
        # Normalize values
        augmented = augmentations(image = img).astype(np.float32) / 255
        return augmented       
  
    def push_data(self, path, train = True):
        # returns a generator for the specified data.
        if train:
            true_path = os.path.join(path, "Train")
        else:
            true_path = os.path.join(path, "Test")

        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function = lambda im: self.preprocess_sequence(im, train))

        data_gen = data_gen.flow_from_directory(
            true_path, 
            target_size=(hp.img_size, hp.img_size),
            class_mode = None,
            batch_size=hp.batch_size)
        return data_gen

