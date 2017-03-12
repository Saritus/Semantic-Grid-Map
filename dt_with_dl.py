#!/usr/bin/env python

import lasagne
from lasagne import layers
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer

from lasagne.layers import Conv2DLayer as Conv2DLayerSlow
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerSlow
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print('Using lasagne.layers (slower)')

from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_conv_weights

import matplotlib.pyplot as pyplot
import numpy as np
from scipy import ndimage
from PIL import Image

PATCHSIZE = 4
FOLDER = "autoencoder4"
SETS = 65536/64

def load_data():
    x_train = []
    y_train = []

    for x in np.arange(0, SETS):
        # Erstellen einer Zufallskarte
        x_train_sample = np.random.choice([0, 1], size=(PATCHSIZE,PATCHSIZE), p=[0.50, 0.50])
        y_train_sample = ndimage.distance_transform_edt(x_train_sample)
        y_train_sample = x_train_sample

        #show_array(x_train_sample.reshape((PATCHSIZE, PATCHSIZE)))
        #show_array(y_train_sample.reshape((PATCHSIZE, PATCHSIZE)))

        x_train.extend([[x_train_sample]])
        y_train.extend([y_train_sample])

    # Umwandeln der beiden Arrays in numpy-Arrays
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    y_train = y_train.reshape(-1, PATCHSIZE*PATCHSIZE)
    #print y_train.shape

    x_valid = x_train
    y_valid = y_train
    x_test = x_train
    y_test = y_train

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def show_array(array, filename="", folder=""):
    pyplot.imshow(array)
    pyplot.show()


def save_array(array, filename, folder=""):
    # Normalize image to [0,255]
    #imagearray = array - array.min()
    #imagearray = imagearray / imagearray.max() * 255
    imagearray = array# - array.min()
    imagearray = imagearray / imagearray.max() * 255

    
    image = Image.fromarray(imagearray).convert('RGB').resize((500,500), Image.NEAREST)
    image.save(folder + "/" + filename + ".tif")


def save_denselayer(net1, layername, filetype="tif", width=110, height=110):
    layer0_values = layers.get_all_param_values(net1.layers_[layername])
    for neuro in range(0, layer0_values[0].shape[1]):
        layer0_1 = [layer0_values[0][i][neuro] for i in range(len(layer0_values[0]))]
        if filetype != "tif":
            layer0_1 = [i * 256 for i in layer0_1]
        layer0_1 = np.asarray(layer0_1)
        layer0_1 = layer0_1.reshape(
            PATCHSIZE,  # first image dimension (vertical)
            PATCHSIZE  # second image dimension (horizontal)
        )
        image = Image.fromarray(layer0_1)
        if filetype == "tif":
            image.resize((500, 500), Image.NEAREST).save(FOLDER + "/" + layername + str(neuro).zfill(4) + '.tif', "TIFF")
        elif filetype == "png":
            image.convert('RGB').resize((500, 500), Image.NEAREST).save(FOLDER + "/" + str(neuro).zfill(4) + '.png', "PNG")
        else:
            sys.stderr.write('Filetype is not supported')


def create_net():
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                #('conv2d1', layers.Conv2DLayer),
                ('hidden1', layers.DenseLayer),
                #('hidden2', layers.DenseLayer),
                #('hidden3', layers.DenseLayer),
                #('hidden4', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None, 1, PATCHSIZE, PATCHSIZE),

        #conv2d1_num_filters=64,
        #conv2d1_filter_size=(7, 7),
        #conv2d1_nonlinearity=lasagne.nonlinearities.identity,
        #conv2d1_W=lasagne.init.GlorotUniform(),

        hidden1_num_units=PATCHSIZE*PATCHSIZE,  # number of units in 'hidden' layer
        #hidden2_num_units=PATCHSIZE*PATCHSIZE*2,  # number of units in 'hidden' layer
        #hidden3_num_units=121,  # number of units in 'hidden' layer
        #hidden4_num_units=5,  # number of units in 'hidden' layer

        output_num_units = PATCHSIZE*PATCHSIZE,
        output_nonlinearity = lasagne.nonlinearities.identity,

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.5,

        regression=True,
        max_epochs=1,
        verbose=1,
    )
    return net1


def main():
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()

    print("Got %i testing datasets." % len(x_train))

    # Create the net
    net1 = create_net()

    scores = []
    net1.fit(x_train, y_train)

    # Train the network
    for i in np.arange(0, 5000):
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()

        #current_score = net1.score(x_test, y_test)
        #print current_score
        #scores.extend([current_score])
        
        net1.fit(x_train, y_train)
    
        #if(i%2 is 0):
    # Show the result that we want and the result that we get
            #for x in np.arange(0, SETS):
                #save_array(y_test[x].reshape((PATCHSIZE, PATCHSIZE)), str(x).zfill(4)+"_t", FOLDER)
                #save_array(net1.predict(x_test)[x].reshape((PATCHSIZE, PATCHSIZE)), str(x).zfill(4)+"_y", FOLDER)
        save_denselayer(net1, "hidden1", "tif", 500, 500)
            
    print scores

    #plot_conv_weights(net1.layers_['conv2d1'], figsize=(7, 7))
    #pyplot.show()

    # Try the network on new data
    #print("Label:\n%s" % str(y_test[:5]))
    #print("Predicted:\n%s" % str(net1.predict(x_test[:5])))


if __name__ == '__main__':
    main()
