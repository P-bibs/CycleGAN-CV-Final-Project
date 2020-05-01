import tensorflow as tf
import hyperparameters as hp 
from tensorflow.keras.layers import \
        Conv2D, LeakyReLU, Conv2DTranspose, Activation, Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from keras.models import Model
from numpy import load, ones, zeros, random, asarray
from numpy.random import randint
import matplotlib.pyplot as plt

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        # define layers here

        init = RandomNormal(stddev=0.02)
        self.architecture = []
        # C64
        self.architecture.append(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.architecture.append(LeakyReLU(alpha=0.2))
        # C128
        self.architecture.append(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(LeakyReLU(alpha=0.2))
        # C256
        self.architecture.append(Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(LeakyReLU(alpha=0.2))
        # C512
        self.architecture.append(Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(LeakyReLU(alpha=0.2))
        # second last output layer
        self.architecture.append(Conv2D(512, (4,4), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(LeakyReLU(alpha=0.2))
        # patch output
        self.architecture.append(Conv2D(1, (4,4), padding='same', kernel_initializer=init))
    
    def call(self, img):
        """ Passes input image through the network. """
        for layer in self.architecture:
            img = layer(img)
        return img

class Generator(tf.keras.Model):

    # generator a resnet block
    def resnet_block(self, n_filters, input_layer):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # first layer convolutional layer
        g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # second convolutional layer
        g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        # concatenate merge channel-wise with input layer
        g = Concatenate()([g, input_layer])
        return g

    def __init__(self):
        super(Generator, self).__init__()
        # define layers here

        init = RandomNormal(stddev=0.02)
        self.architecture = []
        # c7s1-64
        self.architecture.append(Conv2D(64, (7,7), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(Activation('relu'))
        # d128
        self.architecture.append(Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(Activation('relu'))
        # d256
        self.architecture.append(Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(Activation('relu'))
        # R256
        for _ in range(6):
            # first convolutional layer
            self.architecture.append(Conv2D(256, (3,3), strides=(1,1), kernel_initializer=init))
            self.architecture.append(InstanceNormalization(axis=-1))
            self.architecture.append(Activation('relu'))
            # second convolutional layer
            self.architecture.append(Conv2D(256, (3,3), strides=(1,1), kernel_initializer=init))
            self.architecture.append(InstanceNormalization(axis=-1))
        # u128
        self.architecture.append(Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(Activation('relu'))
        # u64
        self.architecture.append(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(Activation('relu'))
        # c7s1-3
        self.architecture.append(Conv2D(3, (7,7), padding='same', kernel_initializer=init))
        self.architecture.append(InstanceNormalization(axis=-1))
        self.architecture.append(Activation('tanh'))
    
    def call(self, img):
        """ Passes input image through the network. """
        for layer in self.architecture:
            img = layer(img)
        return img

# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# ensure the model we're updating is trainable
	g_model_1.trainable = True
	# mark discriminator as not trainable
	d_model.trainable = False
	# mark other generator model as not trainable
	g_model_2.trainable = False
	# discriminator element
	input_gen = tf.keras.Input(shape=(hp.img_size, hp.img_size, 3))
	gen1_out = g_model_1(input_gen) # gen_B
	output_d = d_model(gen1_out) # dec_gen_b
	# identity element
	input_id = tf.keras.Input(shape=(hp.img_size, hp.img_size, 3))
	output_id = g_model_1(input_id)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	# define optimization algorithm configuration
	opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model

# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y
 
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y
 
# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('Saved: %s and %s' % (filename1, filename2))
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# plot real images
	for i in range(n_samples):
		plt.subplot(2, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_in[i])
	# plot translated image
	for i in range(n_samples):
		plt.subplot(2, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_out[i])
	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	plt.savefig(filename1)
	plt.close()
 
# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)
 
# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
	# define properties of the training run
	n_epochs, n_batch, = 50, 1
	# determine the output square shape of the discriminator
	n_patch = d_model_A.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# prepare image pool for fakes
	poolA, poolB = list(), list()
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	min_dA_loss1 = 999
	min_dA_loss2 = 999
	min_dB_loss1 = 999
	min_dB_loss2 = 999
	min_g_loss1 = 999
	min_g_loss2 = 999
	for i in range(n_steps):
		# select a batch of real samples
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		# update fakes from pool
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
		# update generator B->A via adversarial and cycle loss
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# update discriminator for A -> [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		# update generator A->B via adversarial and cycle loss
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# update discriminator for B -> [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		# summarize performance
		print('%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
		if dA_loss1 < min_dA_loss1 or dA_loss2 < min_dA_loss2 or dB_loss1 < min_dB_loss1 or dB_loss2 < min_dB_loss2 or g_loss1 < min_g_loss1 or g_loss2 < min_g_loss2:
			# save the models
			save_models(i, g_model_AtoB, g_model_BtoA)
			min_dA_loss1 = min(dA_loss1, min_dA_loss1)
			min_dA_loss2 = min(dA_loss2, min_dA_loss2)
			min_dB_loss1 = min(dB_loss1, min_dB_loss1)
			min_dB_loss2 = min(dB_loss2, min_dB_loss2)
			min_g_loss1 = min(g_loss1, min_g_loss1)
			min_g_loss2 = min(g_loss2, min_g_loss2)
 
# load image data
dataset = load_real_samples('../data/horse2zebra/horse2zebra_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
#image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = Generator()
g_model_AtoB(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
g_model_AtoB.summary()

# generator: B -> A
g_model_BtoA = Generator()
g_model_BtoA(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
g_model_BtoA.summary()

# discriminator: A -> [real/fake]
d_model_A = Discriminator()
d_model_A(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
d_model_A.summary()
d_model_A.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

# discriminator: B -> [real/fake]
d_model_B = Discriminator()
d_model_B(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
d_model_B.summary()
d_model_B.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, (hp.img_size, hp.img_size, 3))
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, (hp.img_size, hp.img_size, 3))
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
