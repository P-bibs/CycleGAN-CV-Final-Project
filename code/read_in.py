import os
import numpy as np
import random
from skimage.io import imread
from skimage.transform import resize
import imgaug as ia
from imgaug import augmenters as iaa
import time
import hyperparameters as hp

class Datasets():
    def __init__(self, dat_dir):
        self.train_A_path = os.path.join(dat_dir, 'trainA')
        self.train_B_path = os.path.join(dat_dir, 'trainB')
        self.test_A_path = os.path.join(dat_dir, 'testA')
        self.test_B_path = os.path.join(dat_dir, 'testB')

        self.train_A = self.push_data(self.train_A_path, Name = "Dataset A")
        self.train_B = self.push_data(self.train_B_path, Name = "Dataset B")
        self.test_A = self.push_data(self.test_A_path, False)
        self.test_B = self.push_data(self.test_B_path, False)

    def preprocess(self, ims, augment):
        def normalize(batch):
            return batch.astype(np.float32) / float(255)
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
                iaa.CLAHE(to_colorspace='HSV')])
        else:
            augmentations = ia.Sequential([iaa.CLAHE(to_colorspace='HSV')])
        
        augmented = augmentations.augment_images(ims)
        for i in augmented:
            if i.shape !=(hp.img_size, hp.img_size, 3):
                print(i.shape)
        augmented = np.stack(augmented, axis = 0)
        return normalize(augmented)    
  
    def push_data(self, path, Name = None,  train = True):
        # returns a generator for the specified data
        data_generator = custom_data_generator(path, (hp.img_size, hp.img_size), 
            self.preprocess, hp.batch_size, True, 1.5, Name = Name)
            # threshold of 1.5 should limit augmentation time to 15 min per epoch
        datagen = data_generator.datagen()
        return datagen

class custom_data_generator():
    # Note: As configured, this generator passes over incomplete batches. This may be problematic for larger batch sizes

    def __init__(self, path, img_size, preprocess_function, batch_size, augment, aug_threshold, Name = None):
        self.img_size = img_size
        self.path = path
        self.files = os.listdir(path)
        self.preprocess_function = preprocess_function
        self.augment = augment
        self.batch_size = batch_size
        # Parameters for adaptive augmentation
        self.aug_threshold = aug_threshold
        self.aug_time = 0
        self.sample_size = 5
        self.is_named = (Name != None)
        self.name = Name

    def get_batch(self, idx):

        def read_proper(name):
            imarray = imread(os.path.join(self.path, name))
            imarray = resize(imarray, self.img_size, preserve_range = True).astype('uint8')
            if len(imarray.shape)==2: 
                imarray = np.stack([imarray for _ in range(3)], axis = -1)
            elif imarray.shape[2]==1:
                imarray = np.concatenate([imarray for _ in range(3)], axis = 2)
            return imarray
            
        start= time.time()
        batch_x = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [read_proper(file_name) for file_name in batch_x]

        if idx == self.sample_size:
            self.aug_time = 0
        if idx < 2 * self.sample_size:
            self.aug_time += time.time() - start
        return batch_x
    
    def adaptive_augment(self, idx):
        # Selectively augments batchs to keep average augmentation time below a threshold
        # Set threshold to None to turn off
        if self.aug_threshold == None:
            pass
        elif idx == self.sample_size:
            avg_time = self.aug_time / float(self.sample_size * self.batch_size)
            p = self.aug_threshold / avg_time
            if p >= 1:
                p = 1
                self.augment = True
                self.aug_threshold = None
            per = (p*100)
            if self.is_named:
                print("\n"+self.name+": Average augmentation time per image is "+str(round(avg_time, 3))+ " seconds.")
                print("\n"+self.name+": Augmention now set to run for " + str(round(per, 3)) + " percent of batches.")
            else:
                print("\n"+"Average augmentation time per image is "+str(round(avg_time, 3))+ " seconds.")
                print("\n"+"Augmention now set to run for " + str(round(per, 3)) + " percent of batches.")
        elif idx == 2* self.sample_size:
            avg_time = self.aug_time / float(self.sample_size * self.batch_size)
            if self.is_named:
                print("\n"+self.name+": Average augmentation time is now", round(avg_time, 3), "seconds per image.")
            else:
                print("\n"+"Average augmentation time is now", round(avg_time, 3), "seconds per image.")
        elif idx > self.sample_size:
            self.augment = (random.random() < p)

    def datagen(self):
        n = -1
        while True:
            n += 1
            if self.batch_size * (n+1) >= len(self.files): n = 0
            self.adaptive_augment(n)
            out = self.preprocess_function(self.get_batch(n), self.augment)
            yield out