#TODO(michael): change preprocess to keep only alpha or alphanumerics

import string
from collections import Counter

import fuel


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

    counts = counter.most_common(vocabulary_size)

    if min_count:
        pass
        #TODO(michael)

    dictionary = {}
    for index, word_count in enumerate(counts):
        word, _ = word_count
        dictionary[word] = index

    dictionary['<UNK>'] = len(dictionary)

    return dictionary


if __name__ == '__main__':
    from pprint import pprint
    pprint(make_dictionary('shakespeare.txt'))


