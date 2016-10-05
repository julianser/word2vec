#TODO(michael):
# change preprocess to keep only alpha or alphanumerics
# add subsampling
# add negative sampling

import string
from collections import Counter

import fuel
from fuel.transformers import Transformer


table = string.maketrans("","")
def preprocess(s):
    """Remove punctuation and make lowercase"""
    return s.translate(table, string.punctuation).lower()


def make_dictionary(files, vocabulary_size=None, min_count=None):
    """Make dictionary containing words and their counts

    if vocabulary size is specified, return only that many of the most common
    words
    """
    counter = Counter()

    for filename in files:
        with open(filename, 'r') as f:
            for line in f:
                counter.update(preprocess(line.strip()).split())

    counts = counter.most_common(vocabulary_size - 1)

    if min_count:
        pass
        #TODO(michael)

    dictionary = {}
    for index, word_count in enumerate(counts):
        word, _ = word_count
        dictionary[word] = index

    dictionary['<UNK>'] = len(dictionary)

    return dictionary


class SkipGram(Transformer):
    def __init__(self, skip_windows, num_skips, data_stream, target_source='targets',
                 **kwargs):
        if not data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce examples, '
                             'not batches of examples.')
        if len(data_stream.sources) > 1:
            raise ValueError('{} expects only one source'
                             .format(self.__class__.__name__))

        super(SkipGram, self).__init__(data_stream, produces_examples=True, **kwargs)
        self.sources = self.sources + (target_source,)

        self.skip_window = skip_window
        self.num_skips = num_skips

        self.source_word_index = 0
        self.skip_counter = 0
        self.sentence = []

    def get_data(self, request=None):
        if request is not None:
            raise ValueError

        if self.skip_counter > self.num_skips:
            self.source_word_index += 1
            self.skip_counter = 0

            if self.source_word_index >= len(sentence):
                while len(self.sentence) <= 1:
                    self.sentence, = next(self.child_epoch_iterator)

        self.skip_counter += 1
        return (source, target)


