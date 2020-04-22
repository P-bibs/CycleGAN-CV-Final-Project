""" Hyperparamters for Horse-to-Zebra (hz) model"""

"""Preprocess parameters"""
#Pre-process parameters

image_size = 128


# Training parameters

num_epochs = 50

batch_size = 10

learning_rate = 0.0002

# in the paper, this is lambda, but lambda is a reserved word in Python
cycle_consistency_weight = 10

