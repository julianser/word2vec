import lasagne
from stochastic_ops import Stochastic_Op

class StochasticLayer(lasagne.layers.Layer):
    def __init__(self, incoming, estimator='MF', **kwargs):
        super(StochasticLayer, self).__init__(incoming, **kwargs)
        self.estimator = estimator
        self.stochastic_op = Stochastic_Op(estimator=self.estimator)

    def get_output_for(self, input_, **kwargs):
        return self.stochastic_op(input_)
