import theano
import lasagne
from theano import tensor as T
from lasagne import layers as L


class Word2Vec(object):
    def __init__(self, batch_size, query_vocab_size, context_vocab_size,
                 emb_dim_size):
        """
        initialize the train and embed methods
        """
        query_input = T.ivector('query')
        context_output = T.ivector('context')
        learning_rate = T.scalar('learning_rate')
        momentum = T.scalar('momentum')

        embedding, network = self.model(batch_size,
                                        query_input,
                                        query_vocab_size,
                                        context_vocab_size,
                                        emb_dim_size)

        prediction = L.get_output(network)
        loss = self.loss(prediction, context_output)
        params = L.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_movement(loss, params,
                                                    learning_rate=learning_rate,
                                                    momentum=momentum)
        self.train = theano.function(
            [query_input, context_output, learning_rate, momentum],
            loss, updates=updates)

        embedding = L.get_output(embedding)
        self.embed = theano.function([query_input], embedding)


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

    def loss(self, logits, context):
        return lasagne.objectives.binary_crossentropy(logits, context)
