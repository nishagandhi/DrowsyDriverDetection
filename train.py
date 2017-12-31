from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import Dataset
import time
import os
import os.path
import numpy as np


def train(batch_size=4, nb_epoch=10):

	checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', 'lstm' + '-' + 'features' + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
save_best_only=True)


	tb = TensorBoard(log_dir=os.path.join('data', 'logs', 'lstm'))

	early_stopper = EarlyStopping(patience=5)

	timestamp = time.time()

	csv_logger = CSVLogger(os.path.join('data', 'logs', 'lstm' + '-' + 'training-' + \
str(timestamp) + '.log'))

	data = Dataset(
            seq_length=26,
            class_limit=2,
        )

	steps_per_epoch = 4

	X, y = data.get_all_sequences_in_memory('training')

	X_test, y_test = data.get_all_sequences_in_memory('testing')

	rm = ResearchModels(len(data.classes),'lstm',data.seq_length, None)
	print "##################################################"
	#X=X[2:]
	#X_test=X_test[2:]
	X=np.ravel(X)
	X=X.reshape(16,26,-1)
	X_test=np.ravel(X_test)
	X_test=X_test.reshape(3,26,-1)
	#print "X", X[0:10]
	print "X.shape", X.shape
	print "y.shape", y.shape
	print "X_test.shape" ,X_test.shape
	print "y_test.shape" ,y_test.shape
	print "##################################################"

	rm.model.fit(X,y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
			epochs=nb_epoch)
	predictions = rm.model.predict(X_test)
	loss, accuracy = rm.model.evaluate(X_test, y_test)
	
	#print 'Loss:',loss*100,'%'
	
	
	for j in predictions:
		if j[0]>j[1]:
			print "Driver is alert with the confidence of",(j[0]*100),"%"
		else:
			print"Driver is drowsy with the confidence of",(j[1]*100),"%"
			print"Sounding the alarm now...."
			duration = 10  # second
			freq = 440  # Hz
			os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
			
def main():

	# model can be one of lstm, lrcn, mlp, conv_3d, c3d
	    batch_size = 4
	    nb_epoch = 10

	    # Chose images or features and image shape based on network.

	    train(batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
	main()
