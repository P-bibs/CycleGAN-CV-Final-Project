#
# hyperparameters
#
solver = Adam
# CycleGAN requires 2x memory: small batch size
batch_size = 1
learning_rate = .0002 # decay over time
num_epochs = 200
weight_initialization = gaussian
convolution_padding = reflected

#
# pre-process
#
data = read_in(data)

augmented_data = augment(data,
    horizontal_shift, vertical_shift,
    salt_pepper_noise, positional_noise, dropout_noise,
    )
    
#
# model
#

# residual block (see ResNet)
# TODO: understand this better (ask Jake)
R256 = [
    conv2D(size=(3,3), filters=256, stride=1, activation=relu),
    conv2D(size=(3,3), filters=256, stride=1, activation=relu),
]

# suggested architecture for 128x128
generator_architecture = [
    # c7s1-64
    conv2D(size=(7,7), filters=64, stride=1, activation=relu),
    # d128
    conv2D(size=(3,3), filters=128, stride=2, activation=relu),
    # d256
    conv2D(size=(3,3), filters=256, stride=2, activation=relu),
    # six R256 blocks
    R256, R256, R256, R256, R256, R256,
    # u128
    conv2d(size=(3,3), filters=128, stride=(1/2), activation=relu),
    # u64
    conv2d(size=(3,3), filters=64, stride=(1/2), activation=relu),
    # c7s1-3
    conv2D(size=(7,7), filters=3, stride=1, activation=relu),
]

# inpired by PatchGAN
# TODO: understand this better
discriminator architecture = [
    conv2d(size=(4,4), filters=64, stride=2, activation=leaky_relu),
    conv2d(size=(4,4), filters=128, stride=2, activation=leaky_relu),
    conv2d(size=(4,4), filters=256, stride=2, activation=leaky_relu),
    conv2d(size=(4,4), filters=512, stride=2, activation=leaky_relu),
]

#
# loss functions
#

# 'x' comes from source domain, 'y' comes from target domain
# G(x) returns a synthesized element of 'y'
# D_y(y) returns 1 if it thinks y is real and 0 if it thinks y is fake
# G is attempting to minimize loss, D_y is attempting to maximize it
adversarial_loss_x = log(D_y(y)) + log(1-D_y(G(x)))

# do it again but for the other set of discriminator and generator:
# F is another generator which is hopefully the inverse of G
adversarial_loss_y = log(D_x(x)) + log(1-D_x(F(y)))

# Now we have a cycle consistency loss to insure the two generators are inverses of each other
# 
cycle_consistency_loss = vector_distance(F(G(x)) - x) + vector_distance(G(F(x)) - y)

# the total loss is a weighted sum of each component loss
lambda = 10
total_loss = adversarial_loss_x + adversarial_loss_y + lambda*cycle_consistency_loss_x