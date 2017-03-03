#!/usr/bin/env python

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import matplotlib.pyplot as pyplot
import numpy as np
from scipy import ndimage

PATCHSIZE = 11

def load_data():
    x_train = []
    y_train = []

    for x in np.arange(0, 10):
        # Erstellen einer Zufallskarte
        x_train_sample = np.random.choice([0, 1], size=(11,11), p=[1./5, 4./5])
        y_train_sample = ndimage.distance_transform_edt(x_train_sample)

        #show_array(x_train_sample.reshape((PATCHSIZE, PATCHSIZE)))
        #show_array(y_train_sample.reshape((PATCHSIZE, PATCHSIZE)))

        x_train.extend([[x_train_sample]])
        y_train.extend([y_train_sample])

    # Umwandeln der beiden Arrays in numpy-Arrays
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    y_train = y_train.reshape(-1, PATCHSIZE*PATCHSIZE)
    print y_train.shape

    x_valid = x_train
    y_valid = y_train
    x_test = x_train
    y_test = y_train

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def show_array(array):
    pyplot.imshow(array)
    pyplot.show()


def create_net():
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden1', layers.DenseLayer),
                ('hidden2', layers.DenseLayer),
                ('hidden3', layers.DenseLayer),
                #('hidden4', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None, 1, PATCHSIZE, PATCHSIZE),

        hidden1_num_units=121,  # number of units in 'hidden' layer
        hidden2_num_units=121,  # number of units in 'hidden' layer
        hidden3_num_units=121,  # number of units in 'hidden' layer
        #hidden4_num_units=5,  # number of units in 'hidden' layer

        output_num_units = PATCHSIZE*PATCHSIZE,
        output_nonlinearity = lasagne.nonlinearities.identity,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,
        max_epochs=1000,
        verbose=1,
    )
    return net1


def main():
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()

    print("Got %i testing datasets." % len(x_train))

    # Create the net
    net1 = create_net()

    # Train the network
    net1.fit(x_train, y_train)

    # Show the result that we want and the result that we get
    #show_array(y_test[0].reshape((PATCHSIZE, PATCHSIZE)))
    show_array(net1.predict(x_test)[elem].reshape((PATCHSIZE, PATCHSIZE)))

    # Try the network on new data
    #print("Label:\n%s" % str(y_test[:5]))
    #print("Predicted:\n%s" % str(net1.predict(x_test[:5])))

if __name__ == '__main__':
    main()