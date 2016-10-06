#TODO(michael)
# port over saving

import time
import numpy as np

import theano
from theano import tensor as T

import lasagne
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy

from word2vec import Word2VecNormal
from dataset import Dataset


def train(files, batch_size, emb_dim_size, save_dir):
    learning_rate = 0.1
    momentum = 0.9
    num_epochs = 3

    dataset = Dataset(files)
    data_stream = dataset.data_stream
    if save_dir:
        dataset.save_dictionary(save_dir)

    query_input = T.ivector('query')
    context_target = T.ivector('context')
    word2vec = Word2VecNormal(batch_size=batch_size,
                              query_input=query_input,
                              context_vocab_size=dataset.vocab_size,
                              query_vocab_size=dataset.vocab_size,
                              emb_dim_size=emb_dim_size)

    prediction = word2vec.get_output()
    loss = categorical_crossentropy(prediction,
                                    context_target)
    loss = loss.mean()
    params = word2vec.get_all_params()
    updates = nesterov_momentum(loss, params, learning_rate, momentum)

    train = theano.function([query_input, context_target], loss,
                            updates=updates)

    losses = []
    for i, batch in enumerate(data_stream.get_batches(batch_size)):
        queries, contexts = batch
        losses.append(train(queries, contexts))

        if i % 100 == 0:
            print('batch {} mean loss {}'.format(i ,np.mean(losses)))


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
