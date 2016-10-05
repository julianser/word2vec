import time
import numpy as np

import theano
from theano import tensor as T
import lasagne
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy

from word2vec import Word2VecNormal
from dataset_reader import DatasetReader
from minibatcher import Minibatcher

theano.config.compute_test_value = 'warn'

def train(files, batch_size, emb_dim_size, learning_rate=0.1, momentum=0.9, num_epochs=3, save_dir=None):
    reader = DatasetReader(
        files=files,
        macrobatch_size=10000,
        num_processes=3,
        min_frequency=0,
        verbose=False)

    if not reader.is_prepared():
        reader.prepare(save_dir=save_dir)

    minibatcher = Minibatcher(
        batch_size=batch_size,
        dtype="int32",
        num_dims=2)

    batch_rows = minibatcher.get_batch()
    query_input = batch_rows[:,0]
    context_target = batch_rows[:,1]

    # word2vec = Word2VecNormal(batch_size,
                              # query_input=query_input,
                              # context_vocab_size=reader.get_vocab_size(),
                              # query_vocab_size=reader.get_vocab_size(),
                              # emb_dim_size=emb_dim_size)

    # prediction = word2vec.get_output()
    # loss = categorical_crossentropy(prediction,
                                    # context_target)
    # loss = loss.mean()
    # params = word2vec.get_all_params()

    # updates = nesterov_momentum(loss,
                                # params,
                                # learning_rate,
                                # momentum)
    # updates.update(minibatcher.get_updates())

    #train = theano.function([], query_input, updates=minibatcher.get_updates())
    # train = theano.function([], loss,
                            # updates=updates, mode='DebugMode')

    for epoch in range(num_epochs):
        #batches = reader.generate_dataset_parallel()
        batches = reader.generate_dataset_serial()
        for batch_num, batch in enumerate(batches):
            print 'Hello'
            print 'running batch {}'.format(batch_num)
            minibatcher.load_dataset(batch)
            losses = []
            for minibatch_num in range(minibatcher.get_num_batches()):
                print 'running minibatch', batch_num
                query = train()
                print 'query {}   context {}'.format(query, None)

            print('batch {} Mean Loss {}'.format(batch_num,np.mean(losses)))

    # if save_dir:
        # word2vec.save_embedder(save_dir)

def test(files, batch_size, num_epochs=3, save_dir=None):
    print 'Training file directory: ', files
    files = ['/Users/NikBel/MILA/Project/word2vec/Data/shakespeare.txt','/Users/NikBel/MILA/Project/word2vec/Data/shakespeare-2.txt' ]
    reader = DatasetReader(
        files=files,
        macrobatch_size=100,
        num_processes=3,
        min_frequency=0,
        kernel=[1,2,3,4,5,6,7,8,9,10,10,9,8,7,6,5,4,3,2,1],
        verbose=True)

    print 'Reader build !'

    if not reader.is_prepared():
        reader.prepare(save_dir=save_dir)

    print 'Reader prepared !'

    minibatcher = Minibatcher(
        batch_size=batch_size,
        dtype="int32",
        num_dims=2)
    print 'Minibatcher built !'

    print 'Number of epochs: ', num_epochs

    batch_rows = minibatcher.get_batch()
    train = theano.function([], batch_rows, updates=minibatcher.get_updates())

    for epoch in range(num_epochs):
        print 'epoch number: ', epoch
        macrobatches = reader.generate_dataset_serial()
        macrobatch_num = 0
        print macrobatches
        for batch in macrobatches:
            macrobatch_num += 1
            print 'running macrobatch {}'.format(macrobatch_num)
            minibatcher.load_dataset(batch)
            for minibatch_num in range(minibatcher.get_num_batches()):
                #print 'running minibatch', minibatch_num
                query = train()
                print 'query: {} '.format(query)


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
    else:
        raise Exception('Must specify either train or test')




