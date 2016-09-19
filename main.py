import os
import sys

import numpy as np
import tensorflow as tf

from utils import get_mnist
import vae

IMG_DIM = 28
ARCHITECTURE = [IMG_DIM**2,
                500, 500,  # intermediate encoding
                2] # latent 
HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
    }

MAX_ITER = 2000
MAX_EPOCHS = np.inf

LOG_DIR="./log"
METAGRAPH_DIR = "./out"
PLOTS_DIR = "./png"

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data")

def main(to_reload=None):
    mnist = load_mnist()

    if re_load:
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
        print "Loaded"

if __name__ == '__main__':
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
           pass

    try:
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except(IndexError):
        main()


