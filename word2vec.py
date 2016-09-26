from abc import ABCMeta

import theano
import lasagne
from theano import tensor as T
from lasagne import layers as L


class Word2VecBase:
    __metaclass__ = ABCMeta

    def __init__(self, batch_size, query_input, query_vocab_size, context_vocab_size,
                 emb_dim_size):
        """
        initialize the train and embed methods
        """
        self.embedding, self.network = self.model(batch_size,
                                        query_input,
                                        query_vocab_size,
                                        context_vocab_size,
                                        emb_dim_size)

        embedding = L.get_output(embedding)
        self.embed = theano.function([self.query_input], self.embedding)

    @abstractmethod
    def model(self, batch_size, query_input, query_vocab_size,
              context_vocab_size, emb_dim_size):
        raise NotImplementedError

    def get_output(self):
        return L.get_output(self.network)

    def get_all_params(self):
        L.get_all_params(self.network, trainable=True)


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

