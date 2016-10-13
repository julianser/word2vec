import os
from six.moves import cPickle

import theano
import lasagne
from lasagne import layers as L

from stochastic_layer import StochasticLayer


class Word2VecBase:
    def __init__(self, batch_size, query_vocab_size, context_vocab_size,
                 emb_dim_size):
        self.batch_size = batch_size
        self.query_vocab_size = query_vocab_size
        self.context_vocab_size = context_vocab_size
        self.emb_dim_size = emb_dim_size
        self.query = T.ivector('query')
        self.context = T.ivector('context')

    def build_model(self):
        (self.query_network,
         self.context_network) = self.model(self.query,
                                            self.context,
                                            self.batch_size,
                                            self.query_vocab_size,
                                            self.context_vocab_size,
                                            self.emb_dim_size)

        self.query_embedding = L.get_output(self.query_network)
        self.context_embedding = L.get_output(self.context_network)

        self.embed = theano.function([self.query], self.query_embedding)

    def model(self, query, context, batch_size, query_vocab_size,
              context_vocab_size, emb_dim_size):
        raise NotImplementedError

    def sigmoid_loss(self):
        loss_vector = self.query_embedding * self.context_embedding
        loss = loss_vector.sum()
        return theano.tensor.nnet.sigmoid(loss)

    def get_all_params(self):
        return L.get_all_params(self.query_network, trainable=True) + \
            L.get_all_params(self.context_network, trainable=True)

    def save(self, save_dir):
        params = [self.batch_size,
                  self.query_vocab_size,
                  self.context_vocab_size,
                  self.emb_dim_size]
        values = L.get_all_param_values(self.query_network)

        filename = os.path.join(save_dir, 'network_params.save')
        with open(filename, 'wb') as f:
            cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)

        filename = os.path.join(save_dir, 'embedder_values.save')
        with open(filename, 'wb') as f:
            cPickle.dump(values, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def load_params(self, save_dir):
        filename = os.path.join(save_dir, 'network_params.save')
        with open(filename, 'rb') as f:
            params = cPickle.load(f)

        (self.batch_size,
            self.query_vocab_size,
            self.context_vocab_size,
            self.emb_dim_size) = params

    def load_embedder(self, save_dir):
        if not self.query_network:
            raise Exception('Must build model before loading embedding values')

        filename = os.path.join(save_dir, 'embedder_values.save')
        with open(filename, 'rb') as f:
            values = cPickle.load(f)
            L.set_all_param_values(self.query_network, values)


class Word2Vec(Word2VecBase):
    def model(self, query, context, batch_size, query_vocab_size,
              context_vocab_size, emb_dim_size):
        l_query_input = L.InputLayer(shape=(batch_size,),
                                     input_var=query)
        l_query_embed = L.EmbeddingLayer(l_input,
                                   input_size=query_vocab_size,
                                   output_size=emb_dim_size)
        l_context_input = L.InputLayer(shape=(batch_size,),
                                       input_var=context)
        l_context_embed = L.EmbeddingLayer(l_input,
                                   input_size=context_vocab_size,
                                   output_size=emb_dim_size)
        # l_out = L.ElemwiseMergeLayer([l_query_embed, l_context_embed],
                                     # theano.tensor.mul)
        return l_query_embed, l_context_embed


class Word2VecNormal(Word2VecBase):
    def model(self, query_input, batch_size, query_vocab_size,
              context_vocab_size, emb_dim_size):
        l_input = L.InputLayer(shape=(batch_size,),
                               input_var=query_input)
        l_embed = L.EmbeddingLayer(l_input,
                                   input_size=query_vocab_size,
                                   output_size=emb_dim_size)
        l_out = L.DenseLayer(l_embed,
                             num_units=context_vocab_size,
                             nonlinearity=lasagne.nonlinearities.softmax)
        return l_embed, l_out


class Word2VecDiscrete(Word2VecBase):
    def model(self, query_input, batch_size, query_vocab_size,
              context_vocab_size, emb_dim_size):
        l_input = L.InputLayer(shape=(batch_size,),
                               input_var=query_input)
        #l_embed_continuous = L.EmbeddingLayer(l_input,
        #                                      input_size=query_vocab_size,
        #                                      output_size=emb_dim_size)
        l_values_discrete = L.EmbeddingLayer(l_input,
                                             input_size=query_vocab_size,
                                             output_size=3)
        l_probabilities_discrete = L.NonlinearityLayer(
            l_values_discrete,
            nonlinearity=lasagne.nonlinearities.softmax)
        l_embed_discrete = StochasticLayer(l_probabilities_discrete)
        #l_merge = L.ElemwiseSumLayer([l_embed_continuous, l_embed_discrete])
        l_out = L.DenseLayer(l_embed_discrete,
                             num_units=context_vocab_size,
                             nonlinearity=lasagne.nonlinearities.softmax)

        return l_values_discrete, l_out
