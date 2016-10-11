import theano
import hyperparameters
import lasagne
import lasagne.layers as L
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
import stochastic_ops
from stochastic_layer import StochasticLayer

theano.config.exception_verbosity = 'high'

def test_stochastic_layer():
    print('FF-Layer: (Batch_size, n_features)')
    print('Building stochastic layer model')
    l_in = L.InputLayer(shape=(10, 100))
    l_2 = L.DenseLayer(l_in, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
    print('Input Layer shape:  ', L.get_output_shape(l_in))
    print('Dense Layer shape: ', L.get_output_shape(l_2))
    l_stochastic_layer = StochasticLayer(l_2)
    print('Stochastic Layer shape:  ', L.get_output_shape(l_stochastic_layer))
    network_output = L.get_output(l_stochastic_layer)
    print('Building function...')
    function = theano.function([l_in.input_var], network_output)
    test_example = np.ones((10, 100))
    print('Sample output: ', function(test_example))

test_stochastic_layer()




