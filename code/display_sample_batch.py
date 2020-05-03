from PIL import Image 
import numpy as np
import hyperparameters as hp
from read_in import Datasets

# Standalone script to visualize augmented data

data = Datasets("C:\\Users\\dichi\\Documents\\Brown CS\\CSCI 1430\\Git_projects\\CycleGAN-CV-Final-Project\\data\\apple2orange")

# Batches to read
batches = 2
# Layout of sample display
layout = (5, 4)
# Which generator samples to display
config=['trA', 'teA', 'trB', 'teB']

def batch_extractor(set, n):
    # Extract n batches from a set
    options = {"trA":data.train_A, "trB":data.train_B, "teA":data.test_A, "teB":data.test_B}
    total = []
    for _ in range(n):
        total.append(next(options[set]))
    return np.concatenate(total)

def block2pane(batch, shape):
    padded = np.pad(batch, ((0,), (6,), (6,), (0,)))
    im_list = np.split(padded, batch.shape[0])
    for i in range(len(im_list)):
        im_list[i] = np.squeeze(im_list[i], 0)
    rows = []
    for i in range(shape[1]):
       rows.append(np.concatenate(im_list[i* shape[0]: i * shape[0] + shape[0]], 1))
    return (np.concatenate(rows) * 255).astype(np.int8)

for i in config:
    sample = block2pane(batch_extractor(i, batches), layout)
    im = Image.fromarray(sample, 'RGB')
    im.save()
