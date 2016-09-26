from DatasetReader import DatasetReader, default_parse
from Minibatcher import Minibatcher


def main():

	#Create the dataset reader object
	#Pretty much the same as the previous implementation
	reader = DatasetReader(
		files=files,
		directories=directories,
		skip=skip,
		macrobatch_size=macrobatch_size,
		max_queue_size=max_queue_size,
		num_processes=num_processes,
		unigram_dictionary=unigram_dictionary,
		load_dictionary_dir=load_dictionary_dir,
		min_frequency=min_frequency,
		t=t,
		kernel=kernel,
		parse=default_parse,
		verbose=verbose
	)

	#Create the minibatcher object
	minibatcher = Minibatcher(
		batch_size=batch_size,
		dtype="int32",
		num_dims=??
	)

	#TODO: Add the loss function

	#Create the SGD update function
	updates = nesterov_momentum(
		loss, embedder.get_params(), learning_rate, momentum
	)

	#Add the minibatcher update function
	updates.update(minibatcher.get_updates())

	#Create the training function
	train = function([], loss, updates=updates)

	#Train
	for epoch in range(num_epochs):

		#Load macrobatch from the dataset reader
		if read_data_async:
			macrobatches = reader.generate_dataset_parallel()
		else:
			macrobatches = reader.generate_dataset_serial()

		macrobatch_num = 0
		for macrobatch in macrobatches:

			macrobatch_num += 1

			#Insert macrobatch into the minibatcher
			minibatcher.load_dataset(macrobatch)
			losses = []
			for batch_num in range(minibatcher.get_num_batches()):
				losses.append(train())
