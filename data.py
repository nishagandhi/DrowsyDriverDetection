import csv
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from keras.utils import to_categorical

class Dataset():

	def __init__(self, seq_length=26, class_limit=2, image_shape=(56, 24, 3)):
	        self.seq_length = seq_length
	        self.class_limit = class_limit
	        self.sequence_path = os.path.join('data', 'sequences')
	        # Get the data.
	        self.data = self.get_data()
		# Get the classes.
	        self.classes = self.get_classes()

	def get_data(self):
		with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
	        	reader = csv.reader(fin)
            		data = list(reader)

        	return data


	def get_classes(self):
		classes = []
		for item in self.data:
		    if item[1] not in classes:
		        classes.append(item[1])

		# Sort them.
		classes = sorted(classes)

		# Return.
		if self.class_limit is not None:
			return classes[:self.class_limit]
		else:
			return classes

	def get_class_one_hot(self, class_str):
        	# Encode it first.
        	label_encoded = self.classes.index(class_str)

        	# Now one-hot it.
        	label_hot = to_categorical(label_encoded, len(self.classes))

        	assert len(label_hot) == len(self.classes)

		return label_hot

	def get_all_sequences_in_memory(self, train_test):

                #train, test = self.split_train_test()
		#data = train if train_test == 'train' else test

		print"Loading samples into memory for --> ",train_test

		X, y = [], []

		for videos in self.data:
					if(videos[0] == train_test):
						sequence = self.get_extracted_sequence(videos)
						if sequence is None:
							print("Can't find sequence. Did you generate them?")
							raise
						X.append(sequence)
						y.append(self.get_class_one_hot(videos[1]))

        	return np.array(X), np.array(y)


	def get_extracted_sequence(self,video):
		"""Get the saved extracted features."""
	        filename = video[2]
	        path = os.path.join(self.sequence_path, filename + '-' + str(26) + \
	            '-' + 'features' + '.npy')
	        if os.path.isfile(path):
	        	return np.load(path)
	        else:
			return None
