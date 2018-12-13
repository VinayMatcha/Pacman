import numpy as np
def mean_square_error(a,y):
	return 1/2 * (np.sum(np.square(y-a))/y.shape[1])