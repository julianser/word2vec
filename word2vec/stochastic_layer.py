import lasagne
import theano.tensor as T
from stochastic_ops import Stochastic_Op

class StochasticLayer(lasagne.layers.Layer):
    def get_output_for(self, input_, **kwargs):
        return Stochastic_Op(input_)
