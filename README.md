# DrowsyDriverDetection
This is a project implementing Computer Vision and Deep Learning concepts to detect drowsiness of a driver and sound an alarm if drowsy.


Youtube Demo : https://youtu.be/3uMlNuXfNfc



•	Built a model for drowsiness detection of a driver by real-time Eye-Tracking in videos using Haar Cascades and CamShift algorithm.


•	Used the significant features for each video frame extracted by CNN from the final pooling layer to stitch as a sequence of feature vectors for consecutive frames.   


•	This sequence (2048-D) is given as an input to Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN), which predicts the drowsiness of the driver given the video sequence and sounds an alarm in such a case.


•	Optimized network weights by Adam Optimization algorithm.

Technologies used: Python 2.7, OpenCV 3.3.0, Tensorflow, Keras, CNN, RNN, LSTM.

Steps to run this project:

1) Run the run_extract_eyes.sh program to track the eyes for different videos(training data) and to store the patches of the eyes in a folder for every video. (Alert and Drowsy)
2) Use this training data to retrain the CNN model(Inception V3 model).
3) Run extract_features.py to extract the features from the second last layer of the CNN model which is a 2048-d vector and to create a sequence of frames as a single vector to be given as an input to the LSTM which is a part of Recurrent Neural Networks(RNN) 
4) Run data.py and models.py
5) Finally run train.py to get the final predictions for the test sequence of data and the alarm will sound if the model predicts the sequence to be in a drowsy state.
 

