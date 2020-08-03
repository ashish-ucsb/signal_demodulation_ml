import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.draw import line_aa

def visualization_block(time_series):
	"""Visualization block: Converts time series to 2D image"""
	n = len(time_series)
	img = np.zeros((28, 28), dtype=np.uint8)
	x = np.linspace(0, 27, n, endpoint=True, dtype=np.uint8)
	y = (time_series*27).astype(int)
	for i in range(n-1):
	    rr, cc, val = line_aa(y[i], x[i], y[i+1], x[i+1])
	    img[rr, cc] = val * 255
	return np.flip(img, axis=0)

if __name__ == '__main__':

	x = loadmat('data/ook_10p_0cm.mat')['data_10p_0cm'].T
	plt.imshow(visualization_block(x[1]))
	plt.show()


