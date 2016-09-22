'''
 This module essentially wraps TokenMap and Sampler and adds some extra
 functionality
'''

import os
from TokenMap import TokenMap, SILENT, WARN, ERROR, UNK
from Sampler import Sampler


class UnigramDictionary(object):
	"""
	Bundles together a TokenMap and CounterSampler.  Provides a method for
	pruning the vocabulary while keeping the TokenMap and CounterSampler
	in sync with one another.
	"""

	def __init__(self, on_unk=WARN, token_map=None, sampler=None):
		"""
		Create a new UnigramDictionary.  Typical usage provides no
		arguments, but a token_map and counter_sampler can be provided
		to build a UnigramDictionary that comprises them.
		"""
		self.on_unk = on_unk
		self.token_map = token_map
		if token_map is None:
			self.token_map = TokenMap(on_unk=on_unk)

		self.sampler = sampler
		if sampler is None:
			self.sampler = Sampler()

	def sort(self):
		"""Sorts the sampler counter and tokens from token_map"""
		unk_count = self.sampler.counts[0]

		# Get the counts and tokens (skipping the first UNK entry)
		# They are parallel arrays (ith count corresponds to ith token)
		counts = self.sampler.counts[1:]
		tokens = self.token_map.tokens[1:]

		# Zip them together and sort by counts
		token_counts = zip(counts, tokens)
		token_counts.sort(reverse=True)

		# Separate them again
		new_counts = [unk_count]
		new_tokens = ['UNK']
		for count, token in token_counts:
			new_counts.append(count)
			new_tokens.append(token)

		# Rebuild the token_map and counter_sampler on the sorted arrays
		self.token_map = TokenMap(on_unk=self.on_unk, tokens=new_tokens)
		self.sampler = Sampler(counts=new_counts)

	def remove(self, token):
		"""Removes id from TokenMap and Sampler"""
		idx = self.get_id(token)
		self.token_map.remove(token)
		self.sampler.remove(idx)

	def compact(self):
		"""Removes the None's"""
		self.token_map.compact()
		self.sampler.compact()

	def prune(self, min_frequency=5):
		"""
		Remove all tokens that have been observed fewer than min_frequency
		times.  Counts for tokens that are removed are attributed to UNK.
		"""
		counts = []
		tokens = []
		for idx, token in enumerate(self.token_map.tokens):

			# Copy over tokens that have at least min_frequency
			# observations. Also copy over UNK no matter what it's
			# frequency.
			if self.sampler.get_frequency(idx) >= min_frequency or idx == 0 :
				tokens.append(token)
				counts.append(self.get_frequency(idx))

			# Skip tokens that have too little frequency.  Attribute their
			# observations to UNK
			else:
				counts[UNK] += self.get_frequency(idx)

		# Create a new TokenMap and CounterFrequency based on the
		# filtered tokens and their counts
		self.token_map = TokenMap(on_unk=self.on_unk, tokens=tokens)
		self.sampler = Sampler(counts=counts)

	def add(self, token):
		"""
		Add a new token.  If this "token type" (which means this specific
		spelling of a word) has not been seen before, add it to the
		mapping.  Also increment the count for that token type.  Return
		its ID under the token mapping.
		"""

		# Get or create an id for this token
		token_id = self.token_map.add(token)

		# Increment the frequency count
		self.sampler.add(token_id)

		return token_id

	def get_vocab_size(self):
		"""
		Return the number of unique tokens in the token_map.
		"""
		return len(self.token_map)

	def get_num_tokens(self):
		"""
		Return the total number of (non-distinct) tokens observed.
		"""
		return len(self.sampler)

	def __len__(self):
		"""
		Same as get_vocab_size().
		Return the number of unique tokens in the token_map.
		"""
		return self.get_vocab_size()

	def update(self, token_iterable):
		"""Adds multiple tokens to the Sampler and the TokenMap"""
		return [self.add(token) for token in token_iterable]

	def get_id(self, token):
		"""
		Get the id (int) for the corresponding token (string).
		"""
		# Delegate to the underlying token_map.
		return self.token_map.get_id(token)

	def get_ids(self, token_iterable):
		"""
		Get the ids (list of ints) for the corresponding tokens (strings)
		issued by token_iterable.
		"""
		# Delegate to the underlying token map.
		return self.token_map.get_ids(token_iterable)

	def get_token(self, idx):
		"""
		Return token (string) for the corresponding id (int)
		"""
		# Delegate to the underlying token map
		return self.token_map.get_token(idx)

	def get_tokens(self, idx_iterable):
		"""
		Return the tokens (list of strings) for the corresponding ids
		(ints) issued by idx_iterable.
		"""
		# Delegate to the underlying token map.
		return self.token_map.get_tokens(idx_iterable)

	def save(self, savedir):
		"""
		Save the UnigramDictionary to the directory specified.  This saves
		the underlying TokenMap and CounterSampler in the directory
		given (savedir), using the default filenames "token-map.gz" and
		"counter-sampler.gz".
		"""

		# If the directory provided is a file, raise an error
		if os.path.exists(savedir):
			if os.path.isfile(savedir):
				raise IOError(
					'Directory specified for saving UnigramDictionary is a '
					'file.'
				)

		# If the directory provided doesn't exist, make it (this will not
		# make parent directories though).
		else:
			os.mkdir(savedir)

		# Save the TokenMap and CounterSampler by delegating to their
		# save functions.
		self.token_map.save(os.path.join(savedir, 'token-map.gz'))
		self.sampler.save(os.path.join(
			savedir, 'counter-sampler.gz'
		))

	def load(self, loaddir):
		"""
		Load a UnigramDictionary from the specified directory, by
		loading the TokenMap and CounterSampler stored there.  This assumes
		the filenames are 'token-map.gz' and 'counter-sampler.gz'.
		"""
		# Load the TokenMap by delegation to its load function
		self.token_map = TokenMap()
		self.token_map.load(os.path.join(loaddir, 'token-map.gz'))

		# Load the CounterSampler by delegation to its load function
		self.sampler = Sampler()
		self.sampler.load(
			os.path.join(loaddir, 'sampler.gz'))

	def sample(self, shape=None):
		"""
		Draw a sample according to the counter_sampler probability
		"""
		# Delegate to the underlying CounterSampler
		return self.sampler.sample(shape)

	def get_probability(self, token_id):
		"""
		Return the probability associated to token_id.
		"""
		# Delegate to the underlying CounterSampler
		return self.sampler.get_probability(token_id)

	def get_frequency(self, token_id):
		"""
		Return the frequency associated to token_id.
		"""
		# Delegate to the underlying CounterSampler
		return self.sampler.get_frequency(token_id)
