#TODO(michael)
# text data stream to (query, context) pairs

import time
import numpy as np

import theano
from theano import tensor as T

import lasagne
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
import fuel
from fuel.datasets.text import TextFile

from word2vec import Word2VecNormal
import dataset


theano.config.exception_verbosity = 'high'
# theano.config.compute_test_value = 'warn'

def main(files, batch_size, emb_dim_size):
    learning_rate = 0.1
    momentum = 0.9
    num_epochs = 3

    dictionary = dataset.make_dictionary(files)
    vocab_size = len(dictionary)
    text_data = TextFile(files,
                         dictionary,
                         unk_token='<UNK>',
                         bos_token=None,
                         eos_token=None,
                         preprocess=dataset.preprocess)

    query_input = T.ivector('query')
    context_target = T.ivector('context')

    word2vec = Word2VecNormal(batch_size,
                              query_input=query_input,
                              context_vocab_size=vocab_size,
                              query_vocab_size=vocab_size,
                              emb_dim_size=emb_dim_size)

    prediction = word2vec.get_output()
    loss = categorical_crossentropy(prediction,
                                    context_target)
    loss = loss.mean()
    params = word2vec.get_all_params()
    updates = nesterov_momentum(loss, params, learning_rate, momentum)

    train = theano.function([query_input, context_target], loss,
                            updates=updates, mode='DebugMode')


    for data in data_stream.get_epoch_iterator():
        print data
        #batches = reader.generate_dataset_parallel()
        # batches = reader.generate_dataset_serial()
        # import pdb
        # pdb.set_trace()
        # for batch_num, batch in enumerate(batches):
            # minibatcher.load_dataset(batch)
            # losses = []
            # for minibatch_num in range(minibatcher.get_num_batches()):
                # print 'running minibatch', batch_num
                # batch_rows = minibatcher.get_batch()
                # queries = batch_rows[:,0]
                # contexts = batch_rows[:,1]
                # losses.append(train(queries, contexts))
                # losses.append(train())

            # print('batch {} Mean Loss {}'.format(batch_num,np.mean(losses)))

if __name__ == '__main__':
    main(['shakespeare.txt'],
         1000,
         500)


