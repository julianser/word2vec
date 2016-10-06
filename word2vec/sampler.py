"""
Calculates the maximum likelihood probability of single words based on the
frequency within the corpus and enables efficiently sampling
from the distribution.


Assumes that the words have been converted into ids that were assigned
by auto-incrementing from 0 (The consequence if that is not the case isn't dire,
it just means memory would be wasted)
"""

import numpy as np
import gzip
import time


class Sampler(object):
    def __init__(self, counts=[]):
        """
        Create a CounterSampler.  Most common usage is to provide no
        arguments, creating an empty CounterSampler, because CounterSampler
        provides functions to accumulate counts from observations.

        counts:	list of counts.  The index of the count in the counts list
        serves as the id for the outcome having that many counts.
        """

        # Validation
        if not all([isinstance(c, int) for c in counts]):
            raise ValueError('Counts must be an iterable of integers')
        self.counts = list(counts)
        self.sampler_ready = False
        self.total = 0
        self.probabilities = None
        self._np_sample = None
        self._ptr = 0

    def remove(self, idx):
        """Removes ID from the Sampler"""
        self.counts[idx] = None
        self.sampler_ready = False

    def compact(self):
        """
        Remake the conts list, eliminating `None`s which are holes
        left by calls to `remvoe()`.
        """
        self.counts = [c for c in self.counts if c is not None]

    def add(self, idx):
        """Adds ID to WordSampler"""
        self.sampler_ready = False
        if len(self.counts) <= idx:
            add_zeros = [0] * (idx + 1 - len(self.counts))
            self.counts.extend(add_zeros)

        self.counts[idx] += 1

    def update(self, idxs):
        """Adds a list of IDs to WordSampler"""
        for idx in idxs:
            self.add(idx)

    def ensure_prepared(self):
        """Ensures the Sampler is prepared"""
        if self.sampler_ready:
            return

        if len(self.counts) < 1:
            raise Exception(
                'Cannot sample if no counts have been made'
            )

        if not self.sampler_ready:
            self.prepare_sampler()

    def prepare_sampler(self, shape=None):
        """Prepares Sampler"""
        if shape is not None:
            size = np.prod(shape)
        else:
            size = 1

        self.total = np.sum(self.counts)
        self.probabilities = np.array(self.counts) / float(self.total)

        num_to_load = max(self.num_to_load, 2 * size)
        self._np_sample = np.random.choice(
            a=len(self.probabilities), size=num_to_load, p=self.probabilities
        )

        self._ptr = 0

        self.sampler_ready = True

    def get_total_count(self):
        """Gets the total count"""
        self.ensure_prepared()
        return self.total

    def __len__(self):
        self.ensure_prepared()
        return self.total

    num_to_load = 10 ** 5

    def sample(self, shape=None):
        """Returns sample"""
        self.ensure_prepared()

        if shape is not None:
            size = np.prod(shape)
        else:
            size = 1

        self._ptr += size

        if self._ptr >= len(self._np_sample):
            num_to_load = max(self.num_to_load, 2 * size)
            self._np_sample = np.random.choice(
                a=len(self.probabilities), size=num_to_load, p=self.probabilities
            )
            self._ptr = 0
            return self.sample(shape)

        else:
            if shape is not None:
                return self._np_sample[self._ptr - size: self._ptr].reshape(shape)
            else:
                return self._np_sample[self._ptr - size: self._ptr][0]

    def get_frequency(self, idx):
        """
        Return the number of observations for outcome idx.
        """
        return self.counts[idx]

    def get_probability(self, token_id):
        """
        Return the probability associated to token_id.
        """
        self.ensure_prepared()
        return self.probabilities[token_id]

    def save(self, filename):
        """Saves Sampler"""
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'w')
        else:
            f = open(filename, 'w')

        for c in self.counts:
            f.write('%d\n' % c)

    def load(self, filename):
        """Loads Sampler"""
        if filename.endswith('.gz'):
            f = gzip.open(filename)
        else:
            f = open(filename)

        self.counts = [int(c) for c in f.readlines()]
