import theano.tensor as T
import lasagne
import lasagne.layers as L
from lasagne.updates import nesterov_momentum
import numpy as np
import theano
from stochastic_layer import StochasticLayer

def test_stochastic_layer_forward_pass():
    print 'FF-Layer: (Batch_size, n_features)'
    print 'Building stochastic layer model'
    l_in = L.InputLayer(shape=(10, 10))
    l_2 = L.DenseLayer(l_in, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)
    print 'Input Layer shape:  ', L.get_output_shape(l_in)
    print 'Dense Layer shape: ', L.get_output_shape(l_2)
    l_stochastic_layer = StochasticLayer(l_2)
    print 'Stochastic Layer shape:  ', L.get_output_shape(l_stochastic_layer)
    l_out = L.DenseLayer(l_stochastic_layer, num_units=1)
    network_output = L.get_output(l_out)
    print 'Building function...'
    function = theano.function([l_in.input_var], network_output)
    test_example = np.ones((10, 10))
    print 'Sample output: ', function(test_example)

def test_stochastic_layer_network():
    learning_rate = 0.1
    momentum = 0.9
    num_epoch = 10000

    input = T.fmatrix('input')
    output = T.fmatrix('output')
    print 'FF-Layer: (Batch_size, n_features)'
    print 'Building stochastic layer model'
    l_in = L.InputLayer(shape=(10,10), input_var=input)
    l_2 = L.DenseLayer(l_in, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)
    print 'Input Layer shape:  ', L.get_output_shape(l_in)
    print 'Dense Layer shape: ', L.get_output_shape(l_2)
    l_stochastic_layer = StochasticLayer(l_2)
    print 'Stochastic Layer shape:  ', L.get_output_shape(l_stochastic_layer)
    l_out = L.DenseLayer(l_stochastic_layer, num_units=1)
    print 'Final Dense Layer shape: ', L.get_output_shape(l_out)
    network_output = L.get_output(l_out)
    print 'Building loss function...'
    loss = lasagne.objectives.squared_error(network_output, output)
    loss = loss.mean()
    params = L.get_all_params(l_out, trainable=True)
    updates = nesterov_momentum(loss, params, learning_rate, momentum)
    train = theano.function([input, output], loss,
                            updates=updates, allow_input_downcast=True)
    output_fn = theano.function([input], network_output, allow_input_downcast=True)
    test_X = np.ones((10, 10))
    test_Y = np.ones((10, 1))

    losses = []
    for epoch in range(num_epoch):

        losses.append(train(test_X, test_Y))
        print('epoch {} mean loss {}'.format(epoch, np.mean(losses)))
        print('Current Output: ', output_fn(test_X))


#test_stochastic_layer_network()
#test_stochastic_layer_forward_pass()





