"""Models for the network"""
import time
import numpy as np
import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
    LeakyReLU, Activation, Conv2D, MaxPool2D, Dropout, Flatten, Dense, Conv2DTranspose, Concatenate, Input, Layer, InputSpec
from tensorflow.keras.models import Sequential
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.initializers import RandomNormal
from os import listdir
# from numpy import asarray
from tensorflow.keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import load_img
from IPython.display import clear_output
import matplotlib.pyplot as plt
import skimage
import glob
import os

# Combination of:
# https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
# https://github.com/misgod/fast-neural-style-keras/blob/master/layers.py
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        if len(padding) == 2:
            self.top_pad = padding[0]
            self.bottom_pad = padding[0]
            self.left_pad = padding[1]
            self.right_pad = padding[1]
        elif len(padding) == 4:
            self.top_pad = padding[0]
            self.bottom_pad = padding[1]
            self.left_pad = padding[2]
            self.right_pad = padding[3]
        else:
            raise TypeError('`padding` should be tuple of int '
                            'of length 2 or 4, or dict. '
                            'Found: ' + str(padding))
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        if s[1] == None:
            return (None, None, None, s[3])
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        top_pad=self.top_pad
        bottom_pad=self.bottom_pad
        left_pad=self.left_pad
        right_pad=self.right_pad        
        return tf.pad(x, [[0, 0], [left_pad,right_pad], [top_pad,bottom_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        print(config)
        return config
    

def create_generator(input_shape):
    # generator a resnet block
    def resnet_block(n_filters, input_layer, i):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # first layer convolutional layer
        g = input_layer
        g = ReflectionPadding2D(padding=(1, 1))(g)
        g = Conv2D(n_filters, (3,3), strides = 1, kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # print(i, "1", "(None, 32, 32, 256)", g.shape)
        # second convolutional layer
        g = ReflectionPadding2D(padding=(1, 1))(g)
        g = Conv2D(n_filters, (3,3), strides = 1, kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        # print(i, "2", "(None, 32, 32, 256)", g.shape)
        # concatenate merge channel-wise with input layer
        g = Concatenate()([g, input_layer])
        return g

    init = RandomNormal(stddev=0.02)

    inputs = Input(shape=input_shape)
    outputs = inputs
    # c7s1-64
    outputs = ReflectionPadding2D(padding=(3, 3))(outputs)
    outputs = Conv2D(64, (7,7), strides=1, kernel_initializer=init, input_shape=input_shape)(outputs)
    outputs = InstanceNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)
    # print("c7s1-64 (None, 128, 128, 64)", outputs.shape)
    # d128
    outputs = ReflectionPadding2D(padding=(1, 1))(outputs)
    outputs = Conv2D(128, (3,3), strides=2, kernel_initializer=init)(outputs)
    outputs = InstanceNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)
    # print("d128 (None, 64, 64, 128)", outputs.shape)
    # d256
    outputs = ReflectionPadding2D(padding=(1, 1))(outputs)
    outputs = Conv2D(256, (3,3), strides=2, kernel_initializer=init)(outputs)
    outputs = InstanceNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)
    # print("d256 (None, 32, 32, 256)", outputs.shape)
    # R256
        # We use 6 residual blocks for 128 × 128 training images,
        # and 9 residual blocks for 256 × 256 or higher-resolution training images.
    for i in range(6):
        outputs = resnet_block(256, outputs, i)
    # u128
    outputs = Conv2DTranspose(128, (3,3), strides=2, padding='same', kernel_initializer=init)(outputs)
    outputs = InstanceNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)
    # print("u128 (None, 64, 64, 128)", outputs.shape)
    # u64
    outputs = Conv2DTranspose(64, (3,3), strides=2, padding='same', kernel_initializer=init)(outputs)
    outputs = InstanceNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)
    # print("u64 (None, 128, 128, 64)", outputs.shape)
    # c7s1-3
    outputs = ReflectionPadding2D(padding=(3, 3))(outputs)
    outputs = Conv2D(3, (7,7), strides=1, kernel_initializer=init)(outputs)
    outputs = InstanceNormalization(axis=-1)(outputs)
    outputs = Activation('tanh')(outputs)
    # print("c7s1-3 (None, 128, 128, 3)", outputs.shape)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


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

        self.generator_g = create_generator((hp.img_size, hp.img_size, 3))
        self.generator_f = create_generator((hp.img_size, hp.img_size, 3))

        self.discriminator_x = Sequential([
            # C64
            Conv2D(64 , 4, strides=2),
            LeakyReLU(alpha=0.2),
            # C128
            Conv2D(128, 4, strides=2),
            InstanceNormalization(axis=-1),
            LeakyReLU(alpha=0.2),
            # C256
            Conv2D(256, 4, strides=2),
            InstanceNormalization(axis=-1),
            LeakyReLU(alpha=0.2),
            # C512
            Conv2D(512, 4, strides=2),
            InstanceNormalization(axis=-1),
            LeakyReLU(alpha=0.2),
            # After the last layer, we apply a convolution to produce a 1-dimensional output.
            Conv2D(1, 4, strides=1)
            ])
        
        self.discriminator_y = Sequential([
            # C64
            Conv2D(64 , 4, strides=2),
            LeakyReLU(alpha=0.2),
            # C128
            Conv2D(128, 4, strides=2),
            InstanceNormalization(axis=-1),
            LeakyReLU(alpha=0.2),
            # C256
            Conv2D(256, 4, strides=2),
            InstanceNormalization(axis=-1),
            LeakyReLU(alpha=0.2),
            # C512
            Conv2D(512, 4, strides=2),
            InstanceNormalization(axis=-1),
            LeakyReLU(alpha=0.2),
            # After the last layer, we apply a convolution to produce a 1-dimensional output.
            Conv2D(1, 4, strides=1)
            ])

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
    #     return np.asarray(data_list)
    
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

    @tf.function
    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        output = []
        with tf.GradientTape(persistent=True) as tape: # Record operations for automatic differentiation.
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            
            # 1. Get the predictions.
            fake_y = self.generator_g(real_x, training=True) #(1, 128, 128, 3)
            cycled_x = self.generator_f(fake_y, training=True)
            
            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            output = [fake_y, cycled_x, fake_x, cycled_y, same_x, same_y]

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

        return output

    def generate_images(self, model, test_input, epoch, n):
        prediction = model(test_input)
            
        plt.figure(figsize=(12, 12))

        display_list = [np.squeeze(test_input, axis=0), np.squeeze(prediction, axis=0)]
        title1 = 'Input Image at epoch ' + str(epoch) + ' image ' + str(n)
        title2 = 'Predicted Image at epoch ' + str(epoch) + " image " + str(n)
        title = [title1, title2]
        for i in range(2):
            plt.subplot(1, 2, i+1) # (rows, cols)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
        # plt.savefig('test')

    def save_images(self, images, dataset, epoch, n_image):
        # tensorflow.python.framework.ops.EagerTensor
        for i in range(8):
            images[i] = np.squeeze(images[i]* 0.5 + 0.5, axis=0) # (128, 128, 3)
        # Delete the old thumbnails, if there are any
        files = glob.glob('results/' + dataset + '/*.jpg')
        for f in files:
            os.remove(f)
        if not os.path.isdir('results'):
            print('Making results directory.')
            os.mkdir('results')
        if not os.path.isdir('results/' + dataset):
            print('Making ' + dataset + ' directory.')
            os.mkdir('results/' + dataset)
        tf.io.write_file('results/' + dataset + '/' + epoch + '_' + n_image + '_' + 'real_x.jpg', tf.image.encode_png((images[0]*255).astype(np.uint8)))
        tf.io.write_file('results/' + dataset + '/' + epoch + '_' + n_image + '_' + 'real_y.jpg', tf.image.encode_png((images[1]*255).astype(np.uint8)))
        tf.io.write_file('results/' + dataset + '/' + epoch + '_' + n_image + '_' + 'fake_y.jpg', tf.image.encode_png((images[2]*255).astype(np.uint8)))
        tf.io.write_file('results/' + dataset + '/' + epoch + '_' + n_image + '_' + 'cycled_x.jpg', tf.image.encode_png((images[3]*255).astype(np.uint8)))
        tf.io.write_file('results/' + dataset + '/' + epoch + '_' + n_image + '_' + 'fake_x.jpg', tf.image.encode_png((images[4]*255).astype(np.uint8)))
        tf.io.write_file('results/' + dataset + '/' + epoch + '_' + n_image + '_' + 'cycled_y.jpg', tf.image.encode_png((images[5]*255).astype(np.uint8)))
        tf.io.write_file('results/' + dataset + '/' + epoch + '_' + n_image + '_' + 'same_x.jpg', tf.image.encode_png((images[6]*255).astype(np.uint8)))
        tf.io.write_file('results/' + dataset + '/' + epoch + '_' + n_image + '_' + 'same_y.jpg', tf.image.encode_png((images[7]*255).astype(np.uint8)))       

    def train(self, train_x, train_y, dataset):
        # raise Exception("Model training not yet implemented")
        print("Beginning training")

        for epoch in range(hp.num_epochs):

            start = time.time()

            n_image = 0
            for batch_image_x, batch_image_y in zip(train_x, train_y):
                for i in range(hp.batch_size):
                    image_x = np.expand_dims(batch_image_x[i], axis=0)
                    image_y = np.expand_dims(batch_image_y[i], axis=0)
                    images = [image_x, image_y]
                    images =  images + self.train_step(image_x, image_y) # ignoring batch size/update for every image

                    if n_image % 10 == 0:
                        # print ('.', end='')
                        print("Sample image at epoch", epoch + 1, "image", n_image + 1)
                        # self.generate_images(self.generator_g, image_x, epoch + 1, n_image + 1)
                        self.save_images(images, dataset, str(epoch + 1), str(n_image + 1))
                    if n_image >= hp.max_images_per_epoch:
                        break
                    n_image+=1

            clear_output(wait=True)
            # Using a consistent image (sample_image) so that the progress of the model
            # is clearly visible.

            if epoch % 1 == 0: # (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                time.time()-start))

    def test(self, data_generator):
        # TODO: add model evaluation
        raise Exception("Model evaluation not yet implemented")