import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	shape = np.shape(X)
	for i in range(0, shape[0], batch_size):
		if y is not None:
			yield [X[i:i+batch_size].tolist(), y[i:i+batch_size].tolist()]
		else:
			yield X[i:i+batch_size].tolist()