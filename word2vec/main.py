import os
import operator

import lasagne
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
import numpy as np
from scipy import spatial
import theano
from theano import tensor as T

from word2vec import Word2VecNormal
from word2vec import Word2VecDiscrete
from dataset import Dataset


def train(files, batch_size, emb_dim_size, save_dir, load_dir):
    learning_rate = 0.1
    momentum = 0.9
    num_epochs = 3

    dataset = Dataset(files, load_dir=load_dir)
    data_stream = dataset.data_stream
    if save_dir:
        dataset.save_dictionary(save_dir)

    query_input = T.ivector('query')
    context_target = T.ivector('context')
    word2vec = Word2VecDiscrete(batch_size=batch_size,
                                context_vocab_size=dataset.vocab_size,
                                query_vocab_size=dataset.vocab_size,
                                emb_dim_size=emb_dim_size)
    word2vec.build_model(query_input)

    prediction = word2vec.get_output()
    loss = categorical_crossentropy(prediction, context_target)
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

        if save_dir and i % 10000 == 0:
            word2vec.save(save_dir)

    if save_dir:
        word2vec.save(save_dir)

    import pdb
    pdb.set_trace()


def test(load_dir):
    dictionary = np.load(os.path.join(load_dir, 'dictionary.npy'))
    dicts = dictionary.item()

    query_input = T.ivector('query')
    word2vec = Word2VecNormal(None, None, None, None)
    word2vec.load_params(load_dir)
    word2vec.build_model(query_input)
    word2vec.load_embedder(load_dir)

    def embed(word):
        return word2vec.embed([dicts[word]])

    def test_quad(a, b, c, d):
        # a_ = embed(a)
        # b_ = embed(b)
        # c_ = embed(c)
        # query = a_ - b_ + c_
        query = embed(a)
        results = [(word, spatial.distance.euclidean(query, embed(word)))
                    for word in dicts]
        results.sort(key=operator.itemgetter(1))
        out = [r[0] for r in results[:10]]

        print 'closest to {} : {}'.format(a, out)
        # print "{} - {} + {} should = {}, result {}".format(a, b, c, d, out)


    tests = [
        ['king', 'man', 'woman', 'queen'],
        ['countess', 'woman', 'man', 'earl'],
        ['queen', 'woman', 'man', 'king'],
    ]

    for test in tests:
        test_quad(*test)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Word2Vec')
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--file', help='file to train off of')
    parser.add_argument('--batch_size', type=int, default=10, help='size of each training batch')
    parser.add_argument('--embed_size', type=int, default=100, help='size of the embedding dimension')
    parser.add_argument('--save_dir', help='directory where dictionary + embedder are saved to/loaded from')
    parser.add_argument('--load_dir', help='directory where dictionary + embedder are saved to/loaded from')

    args = parser.parse_args()

    if args.mode == 'train' and not args.file:
        raise Exception('Must specify training file if in train mode')

    if args.mode == 'train':
        train([args.file], args.batch_size, args.embed_size,
              save_dir=args.save_dir, load_dir=args.load_dir)
    elif args.mode == 'test':
        test(args.load_dir)
