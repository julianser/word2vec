import time

import theano
from theano import tensor as T
import lasagne
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy

from word2vec import Word2VecNormal
from dataset_reader import DatasetReader
from minibatcher import Minibatcher


def main(files, batch_size, emb_dim_size):
    learning_rate = 0.1
    momentum = 0.9

    reader = DatasetReader(
        files=files,
        macrobatch_size=10000,
        num_processes=3,
        min_frequency=0,
        verbose=True)

    minibatcher = Minibatcher(
        batch_size=batch_size,
        dtype="int32",
        num_dims=1)

    query_input = T.ivector('query')
    context_output = T.ivector('context')

    word2vec = Word2VecNormal(batch_size,
                              query_input,
                              reader.get_vocab_size(),
                              reader.get_vocab_size(),
                              emb_dim_size)

    prediction = word2vec.get_output()
    loss = categorical_crossentropy(prediction,
                               context_output)
    loss = loss.mean()
    params = word2vec.get_all_params()
    updates = nesterov_momentum(loss,
                                params,
                                learning_rate,
                                momentum)
    updates.update(minibatcher.get_updates())

    train = theano.function([query_input, context_output],
                            loss,
                            updates=updates)

    for epoch in range(num_epochs):
        #batches = reader.generate_dataset_parallel()
        batches = reader.generate_dataset_serial()

        for batch_num, batch in enumerate(batches):

            minibatcher.load_dataset(batch)
            train_err = 0
            for minibatch_num in range(minibatcher.get_num_batches()):
                batch_rows = minibatches.get_batch()
                queries = batch_rows[:,0]
                contexts = batch_rows[:,1]
                train_err += train(queries, contexts)

            print('batch {} train error {}'.format(batch_num,
                                                   train_err))

    print word2vec.embed('hello')

if __name__ == '__main__':
    main(['shakespeare.txt'],
         1000,
         500)


