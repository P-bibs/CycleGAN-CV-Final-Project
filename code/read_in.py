
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
import hyperparameters as hp

class Datasets():
    def __init__(self, dat_dir, merge_test = False):
        self.train_A_path = os.path.join(dat_dir, 'trainA')
        self.train_B_path = os.path.join(dat_dir, 'trainB')
        self.test_A_path = os.path.join(dat_dir, 'testA')
        self.test_B_path = os.path.join(dat_dir, 'testB')

        self.merge_test = merge_test
        # If True, train contains both test and train images

        self.train_A = self.push_data(self.train_A_path)
        self.train_B = self.push_data(self.train_B_path)
        self.test_A = self.push_data(self.test_A_path, False)
        self.test_B = self.push_data(self.test_B_path, False)

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

        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function = lambda im: self.preprocess_sequence(im, train))

        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(hp.img_size, hp.img_size),
            class_mode = None,
            batch_size=hp.batch_size)
        return data_gen

class custom_data_generator(Sequence):
    # May be a good replacement for the keras generator above
    # Note: takes image directory directly, no filler file. Run with _iter_

    def __init__(self, path, img_size, preprocess_function, batch_size):
        self.image_size = image_size
        self.files = os.lsdir(path)
        self.preprocess_function = preprocess_function
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(os.path.join(path, file_name)), self.img_size)
               for file_name in batch_x])
