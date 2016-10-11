import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as tensor
import numpy as np
import hyperparameters

class Stochastic_Op(theano.Op):
    nin = 1
    nout = 1
    __props__ = ()

    def __init__(self, estimator):
        super(Stochastic_Op, self).__init__()
        self.rng = MRG_RandomStreams(hyperparameters.RANDOM_SEED)
        self.estimator = estimator

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim not in (1, 2) \
              or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats. Got %s' %
                             x.type)
        return theano.Apply(self,[x,],[x.type()])


    def perform(self, node, input_storage, output_storage, params=None):
        x, = input_storage
        y = self.rng.multinomial(n=1, pvals=x)
        output_storage[0][0] = y.eval()

    def grad(self, inp, grads):
        if self.estimator == 'ST':
            g_sm, = grads
            return [ST_grad_estimator(g_sm)]
        raise NotImplementedError('Estimator Not Implemented.')

    def infer_shape(self, node, shape):
        return shape

    def c_headers(self):
        return ['<iostream>', '<cmath>']

    @staticmethod
    def c_code_template(dtype):
        raise NotImplementedError('C code not implemented')

    def c_code(self, node, name, inp, out, sub):
        raise NotImplementedError('C code not implemented')


Stochastic_Op = Stochastic_Op(estimator=hyperparameters.ESTIMATOR)

class ST_grad_estimator(theano.Op):
    def make_node(self,dy):
        if dy.type.ndim not in (1, 2) \
                or dy.type.dtype not in tensor.float_dtypes:
            raise ValueError('dy must be 1-d or 2-d tensor of floats. Got ',
                             dy.type)
        if dy.ndim == 1:
            dy = tensor.shape_padleft(dy, n_ones=1)
        return theano.Apply(self, [dy], [dy.type()])

    def perform(self, node, input_storage, output_storage, params=None):
        dy, = input_storage
        output_storage[0][0] = dy

    def grad(self):
        raise NotImplementedError('Estimators have no gradient.')

    def infer_shape(self, node, shape):
        return shape

ST_estimator = ST_grad_estimator()
