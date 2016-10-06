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
from fuel.schemes import ShuffledExampleScheme
from fuel.streams import DataStream
from fuel.datasets.text import TextFile

from word2vec import Word2VecNormal
import dataset


theano.config.exception_verbosity = 'high'
# theano.config.compute_test_value = 'warn'

def train(files, batch_size, emb_dim_size, save_dir):
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
    stream = DataStream(text_data)
    data_stream = dataset.SkipGram(skip_window=10,
                                   num_skips=20,
                                   data_stream=stream)


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

    print vocab_size

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
    import argparse
    parser = argparse.ArgumentParser(description='Word2Vec')
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--file', help='file to train off of')
    parser.add_argument('--batch_size', type=int, default=10, help='size of each training batch')
    parser.add_argument('--embed_size', type=int, default=100, help='size of the embedding dimension')
    parser.add_argument('--save_dir', help='directory where dictionary + embedder are saved to/loaded from')

    args = parser.parse_args()

    if args.mode == 'train' and not args.file:
        raise Exception('Must specify training file if in train mode')

    if args.mode == 'train':
        train([args.file], args.batch_size, args.embed_size, save_dir=args.save_dir)
    elif args.mode == 'test':
        test([args.file], args.batch_size, args.embed_size, save_dir=args.save_dir)


