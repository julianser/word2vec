import time

import theano
from theano import tensor as T
import lasagne
from lasagne.updates import nesterov_movement
from lasange.objectives import binary_crossentropy

from word2vec import Word2Vec

def main():
    data = load_data()
    query_train, context_train, query_eval, context_eval = data

    query_input = T.ivector('query')
    context_output = T.ivector('context')

    word2vec = Word2Vec(batch_size,
                        query_input,
                        query_vocab_size,
                        context_vocab_size,
                        emb_dim_size)

    prediction = word2vec.get_output()
    loss = binary_crossentropy(prediction,
                               context_output)
    params = wordd2vec.get_all_params()
    updates = nesterov_movement(loss,
                                params,
                                learning_rate,
                                momentum)
    updates.update(#stuff here)

    train = theano.function([query_input, context_output],
                            loss,
                            updates=updates)


    for epoch in range(num_epochs):
        train_err = 0
        train_batch = 0
        start_time = time.time()

        for batch in minibatches:
            inputs, targets = batch
            train_err +=


