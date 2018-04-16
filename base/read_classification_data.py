# -*- coding: utf-8 -*-
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import os

def load_cifar10_dataset(cifar_dir, mode='supervised'):
    """Load the cifar10 dataset.

    :param cifar_dir: path to the dataset directory
        (cPicle format from: https://www.cs.toronto.edu/~kriz/cifar.html)
    :param mode: 'supervised' or 'unsupervised' mode

    :return: train, test data:
            for (X, y) if 'supervised',
            for (X) if 'unsupervised'
    """
    # Training set
    trX = None
    trY = np.array([])

    # Test set
    teX = np.array([])
    teY = np.array([])

    for fn in os.listdir(cifar_dir):

        if not fn.startswith('batches') and not fn.startswith('readme'):
            fo = open(os.path.join(cifar_dir, fn), 'rb')
            data_batch = pickle.load(fo)
            fo.close()

            if fn.startswith('data'):

                if trX is None:
                    trX = data_batch['data']
                    trY = data_batch['labels']
                else:
                    trX = np.concatenate((trX, data_batch['data']), axis=0)
                    trY = np.concatenate((trY, data_batch['labels']), axis=0)

            if fn.startswith('test'):
                teX = data_batch['data']
                teY = data_batch['labels']

    trX = trX.astype(np.float32) / 255.
    teX = teX.astype(np.float32) / 255.

    if mode == 'supervised':
        return trX, trY, teX, teY

    elif mode == 'unsupervised':
        return trX, teX