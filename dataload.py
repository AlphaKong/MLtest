# -*- coding: utf-8 -*-
import pickle
import gzip

import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open(r'dataset\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='iso-8859-1')
    
    '''
    training_data = list(zip(training_inputs, training_results))
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_data = list(zip(test_inputs, te_d[1]))
    '''
    f.close()
    return (training_data, validation_data, test_data)
