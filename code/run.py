""" CycleGAN final project"""

import os
import argparse
import tensorflow as tf
from model import CycleGANModel
import hyperparameters as hp
from read_in import Datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#TODO: This whole thing needs a do-over. Its mostly just copied from project 4.

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some CycleGANs!")
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['horse-zebra', 'day-night', 'apples-oranges', 'summer-winter'],
        help='''Which dataset to run''')
    parser.add_argument(
        '--data',
        default=os.getcwd() + '/../data/',
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights. ''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}.h5",
            save_best_only=True,
            save_weights_only=False),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(datasets)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
    )

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """
    if ARGS.dataset == "horse-zebra":
        data_dir = "../data/horse2zebra"
    elif ARGS.dataset == "day-night":
        # TODO: get day to night data
        raise Error("Day-night data not yet gathered")
        data_dir = ""
    elif ARGS.dataset == "apples-oranges":
        data_dir = "../data/apple2orange"
    elif ARGS.dataset == "summer-winter":
        data_dir = "../data/summer2winter_yosemite"

    datasets = Datasets(data_dir)

    cycleGAN_model = CycleGANModel()

    if ARGS.evaluate:
        cycleGAN_model.test(datasets.test_A, datasets.test_B)
    else:
        cycleGAN_model.train(datasets.train_A, datasets.train_B)



# Make arguments global
ARGS = parse_args()

main()
