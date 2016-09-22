import time

import theano
from theano import tensor as T

from word2vec import Word2Vec

def main():
    data = load_data()
    query_train, context_train, query_eval, context_eval = data



    for epoch in range(num_epochs):
        train_err = 0
        train_batch = 0
        start_time = time.time()

        for batch in minibatches:
            inputs, targets = batch
            train_err +=


