import time
import numpy as np

import theano
from theano import tensor as T
import lasagne
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy

from word2vec import Word2VecNormal
from dataset_reader import DatasetReader
from minibatcher import Minibatcher


theano.config.exception_verbosity = 'high'
# theano.config.compute_test_value = 'warn'

def main(files, batch_size, emb_dim_size):
    learning_rate = 0.1
    momentum = 0.9
    num_epochs = 3

    reader = DatasetReader(
        files=files,
        macrobatch_size=10000,
        num_processes=3,
        min_frequency=0,
        verbose=True)

    if not reader.is_prepared():
        reader.prepare()

    minibatcher = Minibatcher(
        batch_size=batch_size,
        dtype="int32",
        num_dims=2)
    ### TESTING
    # test_q = np.zeros(reader.get_vocab_size(), dtype=np.int32)
    # test_q[5] = 1
    # query_input.tag.test_value = test_q
    # test_c = np.zeros(reader.get_vocab_size(), dtype=np.int32)
    # test_c[9] = 1
    # context_output.tag.test_value = test_c

    batch = minibatcher.get_batch()
    query_input = batch[:,0]
    context_output = batch[:,1]
    word2vec = Word2VecNormal(batch_size,
                              query_input=query_input,
                              context_vocab_size=reader.get_vocab_size(),
                              query_vocab_size=reader.get_vocab_size(),
                              emb_dim_size=emb_dim_size)

    prediction = word2vec.get_output()
    loss = binary_crossentropy(prediction,
                               context_output)
    loss = loss.mean()
    params = word2vec.get_all_params()

    updates = nesterov_momentum(loss,
                                params,
                                learning_rate,
                                momentum)
    updates.update(minibatcher.get_updates())

    train = theano.function([],
                            loss,
                            updates=updates, mode='DebugMode')

    for epoch in range(num_epochs):
        #batches = reader.generate_dataset_parallel()
        batches = reader.generate_dataset_serial()
        for batch_num, batch in enumerate(batches):
            minibatcher.load_dataset(batch)
            losses = []
            for minibatch_num in range(minibatcher.get_num_batches()):
                print 'running minibatch', batch_num
                losses.append(train())
            print('batch {} Mean Loss {}'.format(batch_num,np.mean(losses)))

    print word2vec.embed('hello')

if __name__ == '__main__':
    main(['shakespeare.txt'],
         1000,
         500)


