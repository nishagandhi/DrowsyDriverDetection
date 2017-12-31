import numpy as np
import os.path
import csv
import glob
import tensorflow as tf
import h5py as h5py
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model

def extractor(image_path):

	with open('/home/tejas/Desktop/Course-Project-CV/output_graph.pb', 'rb') as graph_file:
		graph_def = tf.GraphDef()
        	graph_def.ParseFromString(graph_file.read())
    		tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
	    pooling_tensor = sess.graph.get_tensor_by_name('pool_3:0')
	    image_data = tf.gfile.FastGFile(image_path, 'rb').read()	
	    pooling_features = sess.run(pooling_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
            pooling_features = pooling_features[0]
	    
	return pooling_features

def extract_features():
	with open('data/data_file.csv','r') as f:
		reader = csv.reader(f)
		for videos in reader:
			path = os.path.join('data', 'sequences', videos[2] + '-' + str(26) + \
'-features.npy')
		
			path_frames = os.path.join('data', videos[0], videos[1])
        		filename = videos[2]
			frames = sorted(glob.glob(os.path.join(path_frames, filename + '/*jpg')))
		
			sequence = []
			for image in frames:
				with tf.Graph().as_default():
					features = extractor(image)
					print 'Appending sequence of image:',image,' of the video:',videos				
				  
                   
              		             																																																																																																																																			
					sequence.append(features)

			np.save(path,sequence)
			print 'Sequences saved successfully'						

extract_features()																																																																				
