
import os
import numpy as np
import random
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import multicore
import time
import hyperparameters as hp

class Datasets():
    def __init__(self, dat_dir):
        self.train_A_path = os.path.join(dat_dir, 'trainA')
        self.train_B_path = os.path.join(dat_dir, 'trainB')
        self.test_A_path = os.path.join(dat_dir, 'testA')
        self.test_B_path = os.path.join(dat_dir, 'testB')

        self.train_A = self.push_data(self.train_A_path)
        self.train_B = self.push_data(self.train_B_path)
        self.test_A = self.push_data(self.test_A_path, False)
        self.test_B = self.push_data(self.test_B_path, False)

    def preprocess_sequence(self, ims, augment):
        def normalize(batch, random_state, parents, hooks):
            return batch.astype(np.float32) / 255
        if augment:
            augmentations = iaa.Sequential([
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
                iaa.CLAHE(to_colorspace='HSV'),
                iaa.Lambda(func_images=normalize)])
            augmentations = multicore.Pool(augmentations)
        else:
            augmentations = multicore.Pool([iaa.CLAHE(to_colorspace='HSV'), iaa.Lambda(func_images=normalize)])
        
        augmented = augmentations.imap_batches_unordered(ims)
        return augmented       
  
    def push_data(self, path, train = True):
        # returns a generator for the specified data
        data_generator = custom_data_generator(path, (hp.img_size, hp.img_size), 
            self.preprocess_sequence, hp.batch_size, True, None)
        datagen = data_generator.datagen()
        return datagen

class custom_data_generator():
    # May be a good replacement for the keras generator above
    # Note: takes image directory directly, no filler file. Run with _iter_

    def __init__(self, path, img_size, preprocess_function, batch_size, augment, aug_threshold):
        self.img_size = img_size
        self.path = path
        self.files = os.listdir(path)
        self.preprocess_function = preprocess_function
        self.augment = augment
        self.batch_size = batch_size
        self.length = int(np.ceil(len(self.files) / float(self.batch_size)))
        self.aug_threshold = aug_threshold
        self.aug_time = []

    def get_batch(self, idx):
        start= time.time()
        if self.batch_size * (idx+1) >= self.length:
            batch_x = self.files[idx * self.batch_size:]
            batch_x = [resize(imread(os.path.join(self.path, file_name)), self.img_size)
               for file_name in batch_x]
            batch_x = np.concatenate(batch_x, axis = 0)
        else: 
            batch_x = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = [resize(imread(os.path.join(self.path, file_name)), self.img_size)
               for file_name in batch_x]
            batch_x = np.concatenate(batch_x, axis = 0)

        self.aug_time += start - time.time()
        return batch_x
    
    def adaptive_augment(self, idx):
        # Selectively augments batchs to keep average augmentation time below a threshold
        # Set threshold to None to turn off
        sample_size = 5
        if self.aug_threshold == None:
            pass
        elif idx == sample_size:
            avg_time = sum(self.aug_time) / float(sample_size * self.batch_size)
            p = self.aug_threshold / float(avg_time)
            if p >= 1:
                p = 1
                self.augment = True
                self.aug_threshold = None
            print("Average augmentation time per image is", avg_time, "seconds.")
            print("Augmention now set to run for", p*100, "%% of batches.")
        else:
            self.augment = (random.random() < p)

    def generator(self):
        n = -1
        while True:
            n += 1
            if n == self.length: n = -1
            self.adaptive_augment(n)
            yield self.get_batch(n)

    def datagen(self):
        base = self.generator()
        return self.preprocess_function(base, self.augment)