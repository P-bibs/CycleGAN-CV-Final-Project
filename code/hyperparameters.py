""" Hyperparamters for Horse-to-Zebra (hz) model"""

"""Preprocess parameters"""
#Pre-process parameters

img_size = 128


# Training parameters

num_epochs = 25

batch_size = 5

max_images_per_epoch  = 1000

generator_history_size = 10

learning_rate = 0.0002

# in the paper, this is lambda, but lambda is a reserved word in Python
cycle_consistency_weight = 10

