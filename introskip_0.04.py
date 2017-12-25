#introskip_0.01

import tensorflow as tf
import numpy as np
import sounddevice as sd
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # suppress tensorflow warnings

# define model
# two-layer neural network: relu -> softmax 

X = tf.placeholder(tf.float32, [None,1251]) #batch data

W = tf.Variable(tf.truncated_normal([1251,2000], stddev=0.1)) #weights
b = tf.Variable(tf.truncated_normal([2000], stddev=0.1)) #biases

W2 = tf.Variable(tf.truncated_normal([2000,2], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([2], stddev=0.1))

Y1 = tf.nn.relu(tf.matmul(X,W) + b)
Y2 = tf.nn.softmax(tf.matmul(Y1,W2) + b2)

# load pretrained weights

saver = tf.train.Saver()
sess = tf.Session()

model_name = "./models/1515single_bs20_N1_batches593_e_1000_acc(0.964)"
saver.restore(sess, model_name)

print('\nLoaded weights from %s\n\nListening for intro...' % model_name)

def record(sess, strictness):
	""" Record 0.0567 seconds of system audio and determine if the 
	recording is part of the intro to the episode

	Returns boolean True/False
	
	Inputs
	sess: tensorflow session with trained model
	strictness: variable between 0 to 1 that controls how strictly the 
	neural network (NN) determines whether it has found the intro or not. Used to 
	reduce false positives
	"""

	sd.default.device = 10 # set the device to listen to system audio (machine dependent)
	fs = 44100 # sampling frequency
	chunk = 2500 # samples to record
	rec = sd.rec(chunk, fs, channels=2)[:,0] # record 25/441 seconds
	sd.wait() 

	sample = np.abs(np.fft.rfft(rec).reshape([1,1251])) # transform
	sample = sample/np.max(sample) # normalize

	prediction = sess.run(Y2, feed_dict={X:sample}) # feed to NN
	if prediction[0,1] > strictness: # test if NN prediction meets threshold
		return True
	else:
		return False

delay = .1

# in order to reduce false positives even further it is best to look for consecutive positives

count = 0 # positives counter
consec = 3 # number of consecutive positives to trigger command

while True:
	r = record(sess, .95)
	if r:
		count += 1
		if count >= consec:
			#os.system('skip.ahk') # runs AutoHotkey script that performs the necessary inputs to skip the intro
			count = 0
			print('[%s] skipped'%time.strftime('%H:%M:%S'))
	else:
		count = 0

	time.sleep(delay)