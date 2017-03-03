#!/usr/bin/env python

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import matplotlib.pyplot as pyplot
import numpy as np


def load_data():
    x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    y_train = np.array([0., 0., 0., 1., 0., 0., 0., 1.], dtype=np.float32)

    x_valid = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
    y_valid = np.array([0., 1., 0., 0.], dtype=np.float32)

    # Erstellen zweier leerer Arrays
    x_test = []
    y_test = []

    # Fuellen der Arrays mit den entsprechenden Werten
    for x in np.arange(0, 1.001, 0.01):
        for y in np.arange(0, 1.001, 0.01):
            x_test.extend([[x, y]])
            y_test.extend([1 if (x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)<0.1 else 0]) # 1, wenn x und y groesser 0.5, ansonsten 0

    # Umwandeln der beiden Arrays in numpy-Arrays
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    #x_test = np.array([[0.5, 0.5], [0.8, 0], [0.9, 0.9], [0, 0.8]], dtype=np.float32)
    #y_test = np.array([0, 0, 1, 0], dtype=np.float32)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def show_array(array):
    pyplot.imshow(array)
    pyplot.show()


def create_net():
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden1', layers.DenseLayer),
                #('hidden2', layers.DenseLayer),
                #('hidden3', layers.DenseLayer),
                #('hidden4', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None, 2),

        hidden1_num_units=8,  # number of units in 'hidden' layer
        #hidden2_num_units=5,  # number of units in 'hidden' layer
        #hidden3_num_units=5,  # number of units in 'hidden' layer
        #hidden4_num_units=5,  # number of units in 'hidden' layer

        output_num_units = 1,
        output_nonlinearity = lasagne.nonlinearities.sigmoid,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,
        max_epochs=1000,
        verbose=2,
    )
    return net1


def main():
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()

    print("Got %i testing datasets." % len(x_train))

    # Create the net
    net1 = create_net()

    # Train the network
    net1.fit(x_test, y_test)

    # Show the result that we want and the result that we get
    show_array(y_test.reshape((101, 101)))
    show_array(net1.predict(x_test).reshape((101, 101)))

    # Try the network on new data
    print("Label:\n%s" % str(y_test))
    print("Predicted:\n%s" % str(net1.predict(x_test)))

if __name__ == '__main__':
    main()