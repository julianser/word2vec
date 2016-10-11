import theano
import lasagne.layers as L
import numpy as np
import stochastic_ops
from stochastic_layer import StochasticLayer

def test_stochastic_layer():
    print('FF-Layer: (Batch_size, n_features)')
    print('Building stochastic layer model')
    l_in = L.InputLayer(shape=(100, 100))
    print('Input Layer shape:  ', L.get_output_shape(l_in))
    #l_stochastic_layer = StochasticLayer(l_in)
    #print('Stochastic Layer shape:  ', L.get_output_shape(l_stochastic_layer))
    l_out = L.DenseLayer(l_in, num_units=10)
    print('Output shape: ', L.get_output_shape(l_out))
    network_output = L.get_output(l_out)
    print('Building function...')
    function = theano.function([l_in.input_var], network_output)
    test_example = np.zeros((100, 100))
    print('Sample output: ', function(test_example))

test_stochastic_layer()




