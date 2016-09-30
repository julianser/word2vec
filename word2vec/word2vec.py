import numpy as np
import os

import theano
import lasagne
from theano import tensor as T
from lasagne import layers as L


class Word2VecBase:
    def __init__(self, batch_size, query_input, query_vocab_size, context_vocab_size,
                 emb_dim_size):
        """
        initialize the train and embed methods
        """
        self.embedder, self.network = self.model(batch_size,
                                                 query_input,
                                                 query_vocab_size,
                                                 context_vocab_size,
                                                 emb_dim_size)

        embedding = L.get_output(self.embedder)
        self.embed = theano.function([query_input], embedding)

    def model(self, batch_size, query_input, query_vocab_size,
              context_vocab_size, emb_dim_size):
        raise NotImplementedError

    def get_output(self):
        return L.get_output(self.network)

    def get_all_params(self):
        return L.get_all_params(self.network, trainable=True)

    def save_embedder(self, save_dir):
        values = L.get_all_param_values(self.embedder)
        filename = os.path.join(save_dir, 'embedder.npz')
        np.savez(filename, values=values)


class Word2VecNormal(Word2VecBase):
    def model(self, batch_size, query_input, query_vocab_size,
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


class Word2VecDiscreteContinuous(Word2VecBase):
    def model(self, batch_size, query_input, query_vocab_size,
              context_vocab_size, emb_dim_size):
        l_input = L.InputLayer(shape=(batch_size,),
                               input_var=query_input)
        l_embed_continuous = L.EmbeddingLayer(l_input,
                                   input_size=query_vocab_size,
                                   output_size=emb_dim_size)
        l
        l_out = L.DenseLayer(l_embed,
                             num_units=context_vocab_size,
                             nonlinearity=lasagne.nonlinearities.softmax)
        return l_embed, l_out

