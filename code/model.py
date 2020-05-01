"""Models for the network"""
import time

import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense, Conv2DTranspose, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import matplotlib.pyplot as plt

class CycleGANModel:
    def __init__(self):
        print("Initializing models")
        self.initialize_models()
        print("Initializing loss functions")
        self.initialize_loss_functions()
        print("Initializing optimizers")
        self.initialize_optimizers()
        print("Initializing checkpointer")
        self.initialize_checkpointer()

    def initialize_models(self):
        init = RandomNormal(stddev=0.02)

        # this is only half an R256 layer, so it must be included twice
        R256 = Conv2D(256, 3, strides=1, activation='relu')

        self.generator_g = Sequential()
        # c7s1-64
        self.generator_g.add(Conv2D(64, (7,7), padding='same', kernel_initializer=init))
        self.generator_g.add(InstanceNormalization(axis=-1))
        self.generator_g.add(Activation('relu'))
        # d128
        self.generator_g.add(Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.generator_g.add(InstanceNormalization(axis=-1))
        self.generator_g.add(Activation('relu'))
        # d256
        self.generator_g.add(Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.generator_g.add(InstanceNormalization(axis=-1))
        self.generator_g.add(Activation('relu'))
        # R256
        for _ in range(6):
            # first convolutional layer
            self.generator_g.add(Conv2D(256, (3,3), strides=(1,1), kernel_initializer=init))
            self.generator_g.add(InstanceNormalization(axis=-1))
            self.generator_g.add(Activation('relu'))
            # second convolutional layer
            self.generator_g.add(Conv2D(256, (3,3), strides=(1,1), kernel_initializer=init))
            self.generator_g.add(InstanceNormalization(axis=-1))
        # u128
        self.generator_g.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.generator_g.add(InstanceNormalization(axis=-1))
        self.generator_g.add(Activation('relu'))
        # u64
        self.generator_g.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.generator_g.add(InstanceNormalization(axis=-1))
        self.generator_g.add(Activation('relu'))
        # c7s1-3
        self.generator_g.add(Conv2D(3, (7,7), padding='same', kernel_initializer=init))
        self.generator_g.add(InstanceNormalization(axis=-1))
        self.generator_g.add(Activation('tanh'))

        print(self.generator_g.summary())
        

        self.generator_f = Sequential()
        # c7s1-64
        self.generator_f.add(Conv2D(64, 7, strides=1, activation='relu', input_shape=(hp.img_size,hp.img_size, 3)))
        # d12
        self.generator_f.add(Conv2D(128, 3, strides=2, activation='relu'))
        # d25
        self.generator_f.add(Conv2D(256, 3, strides=2, activation='relu'))
        # six R256 block
        for i in range(6):
            self.generator_f.add(R256)
            self.generator_f.add(R256)
        # u128: upsampling + convolution
        self.generator_f.add(Conv2DTranspose(128, 3, strides=2, activation='relu'))
        # u6
        self.generator_f.add(Conv2DTranspose(64, 3, strides=2, activation='relu'))
        # c7s1-
        self.generator_f.add(Conv2D(3, 7, strides=1, activation='relu'))


        # TODO: if this doesn't work, experiment with relu slope. Documentation is unclear
        self.discriminator_x = Sequential()

        self.discriminator_x.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.discriminator_x.add(LeakyReLU(alpha=0.2))
        # C128
        self.discriminator_x.add(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.discriminator_x.add(InstanceNormalization(axis=-1))
        self.discriminator_x.add(LeakyReLU(alpha=0.2))
        # C256
        self.discriminator_x.add(Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.discriminator_x.add(InstanceNormalization(axis=-1))
        self.discriminator_x.add(LeakyReLU(alpha=0.2))
        # C512
        self.discriminator_x.add(Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.discriminator_x.add(InstanceNormalization(axis=-1))
        self.discriminator_x.add(LeakyReLU(alpha=0.2))
        # second last output layer
        self.discriminator_x.add(Conv2D(512, (4,4), padding='same', kernel_initializer=init))
        self.discriminator_x.add(InstanceNormalization(axis=-1))
        self.discriminator_x.add(LeakyReLU(alpha=0.2))
        # patch output
        self.discriminator_x.add(Conv2D(1, (4,4), padding='same', kernel_initializer=init))

        self.discriminator_y = Sequential()

        self.discriminator_y.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.discriminator_y.add(LeakyReLU(alpha=0.2))
        # C128
        self.discriminator_y.add(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.discriminator_y.add(InstanceNormalization(axis=-1))
        self.discriminator_y.add(LeakyReLU(alpha=0.2))
        # C256
        self.discriminator_y.add(Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.discriminator_y.add(InstanceNormalization(axis=-1))
        self.discriminator_y.add(LeakyReLU(alpha=0.2))
        # C512
        self.discriminator_y.add(Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.discriminator_y.add(InstanceNormalization(axis=-1))
        self.discriminator_y.add(LeakyReLU(alpha=0.2))
        # second last output layer
        self.discriminator_y.add(Conv2D(512, (4,4), padding='same', kernel_initializer=init))
        self.discriminator_y.add(InstanceNormalization(axis=-1))
        self.discriminator_y.add(LeakyReLU(alpha=0.2))
        # patch output
        self.discriminator_y.add(Conv2D(1, (4,4), padding='same', kernel_initializer=init))

    # load all images in a directory into memory
    # def load_images(path, size=(256,256)):
    #     data_list = list()
    #     # enumerate filenames in directory, assume all are images
    #     for filename in listdir(path):
    #         # load and resize the image
    #         pixels = load_img(path + filename, target_size=size)
    #         # convert to numpy array
    #         pixels = img_to_array(pixels)
    #         # store
    #         data_list.append(pixels)
    #     return asarray(data_list)
    
    def initialize_loss_functions(self):
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        def discriminator_loss(real, generated): # adversarial loss for discriminator
            real_loss = loss_obj(tf.ones_like(real), real)

            generated_loss = loss_obj(tf.zeros_like(generated), generated)

            total_disc_loss = real_loss + generated_loss

            return total_disc_loss * 0.5

        def generator_loss(generated): # adversarial loss for generator
            return loss_obj(tf.ones_like(generated), generated)

        def calc_cycle_loss(real_image, cycled_image): # cycle loss
            loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

            return hp.cycle_consistency_weight * loss1

        def identity_loss(real_image, same_image):
            loss = tf.reduce_mean(tf.abs(real_image - same_image))
            return hp.cycle_consistency_weight * 0.5 * loss

        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss
        self.calc_cycle_loss = calc_cycle_loss
        self.identity_loss = identity_loss

    def initialize_optimizers(self):
        # Optimizer that implements the Adam algorithm.
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def initialize_checkpointer(self):
        checkpoint_path = "../checkpoints/train"

        ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                generator_f=self.generator_f,
                                discriminator_x=self.discriminator_x,
                                discriminator_y=self.discriminator_y,
                                generator_g_optimizer=self.generator_g_optimizer,
                                generator_f_optimizer=self.generator_f_optimizer,
                                discriminator_x_optimizer=self.discriminator_x_optimizer,
                                discriminator_y_optimizer=self.discriminator_y_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    # @tf.function
    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape: # Record operations for automatic differentiation.
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            
            # 1. Get the predictions.
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # 2. Calculate the loss
            # adversarial loss for generators
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)
            # cycle loss
            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)
            
            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
        
        # 3. Calculate the gradients for generator and discriminator using backpropagation.
        # target(first arg) will be differentiated against elements in sources (second arg).
        generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                                self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                                self.generator_f.trainable_variables)
            
        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                    self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                    self.discriminator_y.trainable_variables)
            
        # 4. Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                    self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                    self.generator_f.trainable_variables))
        
        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                        self.discriminator_x.trainable_variables))
        
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                        self.discriminator_y.trainable_variables))

    def generate_images(self, model, test_input):
        prediction = model(test_input)
            
        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1) # (rows, cols)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()


    def train(self, data_generator_x, data_generator_y):
        # raise Exception("Model training not yet implemented")
        sample_image = next(data_generator_x)

        for epoch in range(hp.num_epochs):
            start = time.time()

            # Using a consistent image (sample_image) so that the progress of the model
            # is clearly visible.
            self.generate_images(self.generator_g, sample_image)

            n = 0
            for image_x, image_y in zip(data_generator_x, data_generator_y):
                self.train_step(image_x, image_y)
                if n % 10 == 0:
                    print ('.', end='')
                n+=1

            

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                time.time()-start))

    def test(self, data_generator):
        # TODO: add model evaluation
        raise Error("Model evaluation not yet implemented")