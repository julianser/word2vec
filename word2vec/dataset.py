# TODO(michael):
# change preprocess to keep only alpha or alphanumerics
# add subsampling
# add negative sampling
# add minimum count

from collections import Counter
import os
import string
import random

import fuel
from fuel.datasets.text import TextFile
from fuel.transformers import Transformer
from fuel.streams import DataStream
import numpy as np


table = string.maketrans("", "")

class Dataset:
    def __init__(self, files, vocabulary_size=None, min_count=None,
                 load_dir=None):
        if load_dir is not None:
            self.dictionary = self.load_dictionary(load_dir)
        else:
            if vocabulary_size is not None:
                dictionary_vocab = vocabulary_size - 1
            else:
                dictionary_vocab = None

            self.dictionary = self.make_dictionary(files,
                                                   dictionary_vocab,
                                                   min_count)

        self.vocab_size = len(self.dictionary)

        text_data = TextFile(files,
                             self.dictionary,
                             unk_token='<UNK>',
                             bos_token=None,
                             eos_token=None,
                             preprocess=self._preprocess)
        stream = DataStream(text_data)
        self.data_stream = SkipGram(skip_window=10,
                                    num_skips=20,
                                    data_stream=stream)

    def _preprocess(self, s):
        """Remove punctuation and make string lowercase"""
        return s.translate(table, string.punctuation).lower()

    def make_dictionary(self, files, vocabulary_size=None, min_count=None):
        """Make dictionary containing words and their counts

        if vocabulary size is specified, return only that many of the most common
        words
        """
        counter = Counter()

        for filename in files:
            with open(filename, 'r') as f:
                for line in f:
                    counter.update(self._preprocess(line.strip()).split())

        # TODO(michael): should this be -1?
        counts = counter.most_common(vocabulary_size)

        if min_count:
            pass

        dictionary = {}
        for index, word_count in enumerate(counts):
            word, _ = word_count
            dictionary[word] = index

        dictionary['<UNK>'] = len(dictionary)

        return dictionary

    def save_dictionary(self, save_dir):
        np.save(os.path.join(save_dir, 'dictionary.npy'), self.dictionary)

    def load_dictionary(self, load_dir):
        return np.load(os.path.join(load_dir, 'dictionary.npy')).item()


class SkipGram(Transformer):
    def __init__(self, skip_window, num_skips, data_stream,
                 target_source='targets', **kwargs):
        if not data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce examples, '
                             'not batches of examples.')
        if len(data_stream.sources) > 1:
            raise ValueError('{} expects only one source'
                             .format(self.__class__.__name__))

        super(SkipGram, self).__init__(data_stream, produces_examples=True,
                                       **kwargs)
        self.sources = self.sources + (target_source,)

        self.skip_window = skip_window
        self.num_skips = num_skips

        self.source_index = 0
        self.target_indices = []
        self.skip_counter = 0
        self.sentence = []

    def get_data(self, request=None):
        if request is not None:
            raise ValueError

        if not self.target_indices or self.skip_counter > self.num_skips:
            self.skip_counter = 0

            if self.source_index < len(self.sentence) - 1:
                self.source_index += 1
            else:
                # choose a new sentence with length > 1
                self.sentence = []
                while len(self.sentence) <= 1:
                    self.sentence, = next(self.child_epoch_iterator)

                self.source_index = 0

            # create list of possible target indices
            min_index = max(self.source_index - self.skip_window, 0)
            max_index = min(self.source_index + self.skip_window + 1,
                            len(self.sentence) - 1)
            self.target_indices = range(min_index, self.source_index) +  \
                range(self.source_index+1, max_index+1)

            # random.shuffle(self.target_indices)

        self.target_index = self.target_indices.pop()
        self.skip_counter += 1
        source = self.sentence[self.source_index]
        target = self.sentence[self.target_index]
        return (source, target)

    def get_batches(self, batch_size):
        data = self.get_epoch_iterator()
        while True:
            batch_source = []
            batch_target = []
            try:
                for i in range(batch_size):
                    source, target = next(data)
                    batch_source.append(source)
                    batch_target.append(target)
            except StopIteration:
                break

            yield (batch_source, batch_target)
