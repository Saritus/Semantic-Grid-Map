#!/usr/bin/env python

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_conv_weights

import matplotlib.pyplot as pyplot
import numpy as np
from scipy import ndimage
from PIL import Image

PATCHSIZE = 51

def load_data():
    x_train = []
    y_train = []

    for x in np.arange(0, 1000):
        # Erstellen einer Zufallskarte
        x_train_sample = np.random.choice([0, 1], size=(PATCHSIZE,PATCHSIZE), p=[0.01, 0.99])
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


def save_array(array, filename, folder=""):
    imagearray = array - array.min()
    imagearray = imagearray / imagearray.max() * 256
    image = Image.fromarray(imagearray).convert('RGB')
    image.save(folder + "/" + filename + ".tif")


def create_net():
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                #('conv2d1', layers.Conv2DLayer),
                ('hidden1', layers.DenseLayer),
                ('hidden2', layers.DenseLayer),
                ('hidden3', layers.DenseLayer),
                #('hidden4', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None, 1, PATCHSIZE, PATCHSIZE),

        #conv2d1_num_filters=64,
        #conv2d1_filter_size=(7, 7),
        #conv2d1_nonlinearity=lasagne.nonlinearities.identity,
        #conv2d1_W=lasagne.init.GlorotUniform(),

        hidden1_num_units=121,  # number of units in 'hidden' layer
        hidden2_num_units=121,  # number of units in 'hidden' layer
        hidden3_num_units=121,  # number of units in 'hidden' layer
        #hidden4_num_units=5,  # number of units in 'hidden' layer

        output_num_units = PATCHSIZE*PATCHSIZE,
        output_nonlinearity = lasagne.nonlinearities.identity,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.95,

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
    for x in np.arange(0, 100):
        save_array(y_test[x].reshape((PATCHSIZE, PATCHSIZE)), str(x).zfill(4)+"_t", "dt_ae")
        save_array(net1.predict(x_test)[x].reshape((PATCHSIZE, PATCHSIZE)), str(x).zfill(4)+"_y", "dt_ae")

    #plot_conv_weights(net1.layers_['conv2d1'], figsize=(7, 7))
    #pyplot.show()

    #print net1.score(x_test, y_test)

    # Try the network on new data
    #print("Label:\n%s" % str(y_test[:5]))
    #print("Predicted:\n%s" % str(net1.predict(x_test[:5])))

if __name__ == '__main__':
    main()