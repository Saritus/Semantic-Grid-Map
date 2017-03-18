import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.updates import adam

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import ReshapeLayer
from lasagne.nonlinearities import softmax

from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_conv_weights

import sklearn.metrics

import sys
import os
import gzip
import pickle
import numpy
import math
import matplotlib.pyplot as plt

from pylab import imshow, show, cm
from PIL import Image

from urllib import urlretrieve
import threading, time
import random

from nolearn.lasagne import BatchIterator

class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if isinstance(ds, int):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        #ds = self.ds
        #input_shape = input.shape
        #output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)


# <codecell>

### when we load the batches to input to the neural network, we randomly / flip
### rotate the images, to artificially increase the size of the training set

class FlipBatchIterator(BatchIterator):

    def transform(self, X1, X2):
        X1b, X2b = super(FlipBatchIterator, self).transform(X1, X2)
        X2b = X2b.reshape(X1b.shape)

        bs = X1b.shape[0]
        h_indices = numpy.random.choice(bs, bs / 2, replace=False)  # horizontal flip
        v_indices = numpy.random.choice(bs, bs / 2, replace=False)  # vertical flip

        ###  uncomment these lines if you want to include rotations (images must be square)  ###
        #r_indices = np.random.choice(bs, bs / 2, replace=False) # 90 degree rotation
        for X in (X1b, X2b):
            X[h_indices] = X[h_indices, :, :, ::-1]
            X[v_indices] = X[v_indices, :, ::-1, :]
            #X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3)
        shape = X2b.shape
        X2b = X2b.reshape((shape[0], -1))

        return X1b, X2b


def pickle_load(f):
    return pickle.load(f)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def show_image(x, width=11, height=11):
    x = x.reshape(
        width,  # first image dimension (vertical)
        height,  # second image dimension (horizontal)
    )
    image = Image.fromarray(x)
    imshow(image, cmap=cm.gray)
    show()


def label_to_color(array, color_table):
    print "color table"
    print color_table[0]
    print color_table[1]
    print len(color_table)
    imagearray = []
    for i in range(len(array)):
        imagearray.append(color_table[array[i]])
    imagearray = numpy.array(imagearray, dtype='uint8')
    return imagearray

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = "pkl/only_walls.pkl"
PATCHSIZE = 11

savedir = "walls_conv_restricted"
if not os.path.exists(savedir):
    os.makedirs(savedir)


def _load_dataGZ(url=DATA_URL, filename=DATA_FILENAME):
    """Load data from `url` and store the result in `filename`."""
    if not os.path.exists(filename):
        print("Downloading MNIST dataset")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f)


def _load_data(filename=DATA_FILENAME):
    with open(filename, 'rb') as f:
        return pickle_load(f)


def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = _load_data()
    x_train, y_train = data[0]
    x_valid, y_valid = data[0]
    x_test, y_test = data[1]

    x_train = numpy.array(x_train, dtype='float32')
    y_train = numpy.array(y_train, dtype='float32')
    x_valid = numpy.array(x_valid, dtype='float32')
    y_valid = numpy.array(y_valid, dtype='float32')
    x_test = numpy.array(x_test, dtype='float32')
    y_test = numpy.array(y_test, dtype='float32')

    x_train = x_train.reshape((-1, 1, PATCHSIZE, PATCHSIZE))
    x_valid = x_valid.reshape((-1, 1, PATCHSIZE, PATCHSIZE))
    x_test = x_test.reshape((-1, 1, PATCHSIZE, PATCHSIZE))
    x_ae = x_train.reshape((x_train.shape[0], -1))

    print x_train.shape
    print x_ae.shape

    y_train = numpy.asarray(y_train, dtype=numpy.int32)
    y_valid = numpy.asarray(y_valid, dtype=numpy.int32)
    y_test = numpy.asarray(y_test, dtype=numpy.int32)
    #x_ae = numpy.asarray(x_ae, dtype=numpy.float32)

    return dict(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
        x_ae=x_ae,
        num_examples_train=x_train.shape[0],
        num_examples_valid=x_valid.shape[0],
        num_examples_test=x_test.shape[0],
        input_dim=x_train.shape[1],
        output_dim=len(data[2]),
        color_table=data[2]
    )

data = load_data()


def create_mlp():
    net = NeuralNet(
        layers=[('input', InputLayer),
            #('reshape', FlattenLayer),
            ('hidden1', DenseLayer),
            #('dropout1',layers.DropoutLayer), 
            #('hidden2', DenseLayer),
            #('dropout2',layers.DropoutLayer), 
            #('hidden3', DenseLayer),
            ('output', DenseLayer),
        ],
                
        # layer parameters:
        input_shape=(None, 1, PATCHSIZE, PATCHSIZE),
        #reshape_shape=(([0], 11*11)),outdim
        #reshape_outdim=1,
        hidden1_num_units=121,  # number of units in 'hidden' layer
        #hidden2_num_units=50,  # number of units in 'hidden' layer
        #hidden3_num_units=2601,  # number of units in 'hidden' layer

        #dropout1_p=0.15,
        #dropout2_p=0.15,
                
        output_nonlinearity=softmax,
        output_num_units=data['output_dim'],  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=adam,
        update_learning_rate=0.01,
        #update_momentum=0.9,

        max_epochs=1,
        verbose=1,
    )
    return net


def create_cnn():
    net = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                #('maxpool1', layers.MaxPool2DLayer),
                #('conv2d2', layers.Conv2DLayer),
                #('maxpool2', layers.MaxPool2DLayer),
                #('dropout1', layers.DropoutLayer),
                #('dense', layers.DenseLayer),
                #('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, PATCHSIZE, PATCHSIZE),
        # layer conv2d1
        conv2d1_num_filters=256,
        conv2d1_filter_size=(9, 9),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),  
        # layer maxpool1
        #maxpool1_pool_size=(2, 2),
        # layer conv2d2
        #conv2d2_num_filters=64,
        #conv2d2_filter_size=(5, 5),
        #conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        #maxpool2_pool_size=(2, 2),
        # dropout1
        #dropout1_p=0.1,    
        # dense
        #dense_num_units=256,
        #dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        #dropout2_p=0.1,    
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=data['output_dim']+1,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=1,
        verbose=1
    )
    return net


def create_ae():
    conv_filters = 8
    deconv_filters = 1
    filter_sizes = 1
    ae = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            #('conv', layers.Conv2DLayer),
            #('pool', layers.MaxPool2DLayer),
            #('flatten', ReshapeLayer),  # output_dense
            #('encode_layer', layers.DenseLayer),
            ('hidden', layers.DenseLayer),  # output_dense
            #('unflatten', ReshapeLayer),
            #('unpool', Unpool2DLayer),
            #('deconv', layers.Conv2DLayer),
            ('output_layer', DenseLayer),
        ],
        input_shape=(None, 1, PATCHSIZE, PATCHSIZE),

        #conv_num_filters=conv_filters,
        #conv_filter_size = (filter_sizes, filter_sizes),
        #conv_nonlinearity=None,

        #pool_pool_size=(2, 2),

        #flatten_shape=([0], -1), # not sure if necessary?

        #encode_layer_num_units = 121,

        #hidden_num_units = deconv_filters * (PATCHSIZE + filter_sizes - 1) ** 2 / 4,
        hidden_num_units = 121,

        #unflatten_shape=([0], deconv_filters, 11, 11 ),

        #unpool_ds=(2, 2),

        #deconv_num_filters = deconv_filters,
        #deconv_filter_size = (filter_sizes, filter_sizes),
        #deconv_nonlinearity=None,

        #output_layer_shape =([0], 121),
        output_layer_num_units = 121,

        update_learning_rate = 0.01,
        update_momentum = 0.975,
        #batch_iterator_train=FlipBatchIterator(batch_size=128),
        regression=True,
        max_epochs= 1,
        verbose=1,
    )
    return ae

net1 = create_cnn()


def predict(width, height):
    # Try the network on new data
    #print len(data['x_test'])
    prediction = []
    #for X in data['x_test']:
    #predict.extend(net1.predict([X])[0])
    for i in range(len(data['x_test'])):
        if data['y_test'][i]==1:
            prediction.extend([1])
        else:
            prediction.extend(net1.predict([data['x_test'][i]]))

    imagearray = label_to_color(prediction, data['color_table'])

    imagearray = imagearray.reshape(
        width,  # first image dimension (vertical)
        height,  # second image dimension (horizontal)
        3 # rgb
    )
    image = Image.fromarray(imagearray).transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
    return image


def save_denselayer(layername, filetype="tif", width=110, height=110):
    layer0_values = layers.get_all_param_values(net1.layers_[layername])
    for neuro in range(0, layer0_values[0].shape[1]):
        layer0_1 = [layer0_values[0][i][neuro] for i in range(len(layer0_values[0]))]
        if filetype != "tif":
            layer0_1 = [i * 256 for i in layer0_1]
        layer0_1 = numpy.asarray(layer0_1)
        layer0_1 = layer0_1.reshape(
            11,  # first image dimension (vertical)
            11  # second image dimension (horizontal)
        )
        image = Image.fromarray(layer0_1)
        if filetype == "tif":
            image.resize((width, height), Image.NEAREST).save(savedir + "/" + layername + str(neuro).zfill(4) + '.tif', "TIFF")
        elif filetype == "png":
            image.convert('RGB').resize((width, height), Image.NEAREST).save(savedir + "/" + str(neuro).zfill(4) + '.png', "PNG")
        else:
            sys.stderr.write('Filetype is not supported')


def show_conv2dlayer(layername, filename=None):
    plot_conv_weights(net1.layers_[layername], figsize=(11, 11))
    if filename:
        plt.savefig(savedir+'/'+filename)
        #plt.clf()
    plt.show()


def save_score(filename):
    # Save score to file
    with open(savedir + '/' + filename, 'a') as filept:
        filept.write(str(net1.score(data['x_train'], data['y_train'])) + "\n")


def confusion_matrix():
    preds = net1.predict(data['x_test'])
    colorm = sklearn.metrics.confusion_matrix(data['y_test'], preds)
    plt.matshow(colorm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def show_output(layername):
    instance = data['x_train'][260].reshape(1, 1, 11, 11)
    pred = layers.get_output(net1.layers_[layername], instance, deterministic=False)
    N = pred.shape[1]
    #print pred[0][0].eval()
    print N
    #plt.bar(range(N), pred.ravel())
    plt.show()


def restrict_layer(layername, minimum=None, maximum=None):
    layer_values = layers.get_all_param_values(net1.layers_[layername])
    for to_neuron in range(0, layer_values[0].shape[0]):
        if minimum:
            layer_values[0][to_neuron][layer_values[0][to_neuron] < minimum] = minimum
        if maximum:
            layer_values[0][to_neuron][layer_values[0][to_neuron] > maximum] = maximum

    layers.set_all_param_values(net1.layers_[layername], layer_values)


def shuffle_data():
    # Randomize new order
    length = len(data['x_train'])
    new_order = range(0, length)
    numpy.random.shuffle(new_order)

    # Shuffle x_train
    data_x = [data['x_train'][i] for i in new_order]
    data_x = numpy.asarray(data_x)
    data['x_train'] = data_x

    # Shuffle y_train
    data_y = [data['y_train'][i] for i in new_order]
    data_y = numpy.asarray(data_y)
    data['y_train'] = data_y

    # Shuffle x_ae
    data_ae = [data['x_ae'][i] for i in new_order]
    data_ae = numpy.asarray(data_ae)
    data['x_ae'] = data_ae


def main():

    print("Got %i testing datasets." % len(data['x_train']))

    for prints in range(0, 10):
        for x in range(0, 100):
            shuffle_data()
            # Train the network
            net1.fit(data['x_train'], data['y_train'])

            restrict_conv_layer()
            #restrict_layer("encode_layer", 0.01)
            #restrict_layer(2, 0)


        show_conv2dlayer('conv2d1', '9x9')

    #save_denselayer('encode_layer')
    #save_denselayer('hidden')

    '''
    for i in range(1, 12770):
        result = net1.predict([data['x_test'][i]])
        #print str(i) + " - " + str(result.max())
        if result.max() > 0.2:
            print i
            #print result.shape
            print result
            imagearray = numpy.array([value * 256 for value in result])
            show_image(data['x_test'][i])
            show_image(imagearray)
            #show_image(data['x_ae'][i])
    '''

    '''
    #imshow(image, cmap=cm.gray)
    #show()
    #image.save("epochs" + str(net1.max_epochs).zfill(4) + "_hidden_" + str(net1.hidden1_num_units).zfill(4) +'_'+str(net1.hidden2_num_units).zfill(4)+'_'+str(net1.hidden3_num_units).zfill(4)+ '.png',"PNG")

    image.save(savedir + "/epoch" +str(x * net1.max_epochs).zfill(4) + '.png',"PNG")
    #print savedir + "/epoch" +str(x * net1.max_epochs).zfill(4) + '.png saved'
    '''

    # print net1
    # print net1.score(data['x_train'], data['y_train'])
    # numpy.set_printoptions(threshold=numpy.inf)


def restrict_conv_layer():
    layer_values = layers.get_all_param_values(net1.layers_['conv2d1'])

    for to_neuron in range(0, layer_values[0].shape[0]):
        #layer_values[0][to_neuron][ layer_values[0][to_neuron] < -1] = -1
        layer_values[0][to_neuron][layer_values[0][to_neuron] > 0] = 0

    layers.set_all_param_values(net1.layers_['conv2d1'], layer_values)


if __name__ == '__main__':
    main()
