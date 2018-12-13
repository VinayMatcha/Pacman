import numpy as np

def mnist():
	tr_x = np.load("MNIST_data/train-images.dat")
	tr_y = np.load("MNIST_data/train-labels.dat")
	tv_x = np.load("MNIST_data/validation-images.dat")
	tv_y = np.load("MNIST_data/validation-labels.dat")
	ts_x = np.load("MNIST_data/test-images.dat")
	ts_y = np.load("MNIST_data/test-labels.dat")
	return tr_x,tr_y,tv_x,tv_y,ts_x,ts_y
	

