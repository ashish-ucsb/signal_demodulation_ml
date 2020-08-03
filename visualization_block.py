import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat

def visualization_block(time_series, linewidth=10, output_shape=(28,28)):
	"""Visualization block: Converts time series to 2D image"""
	fig = plt.figure()
	fig.add_subplot(111)
	fig.tight_layout(pad=0)
	plt.ylim(0, 1)
	plt.axis('off')
	plt.plot(time_series, linewidth=linewidth)
	fig.canvas.draw()
	plt.close(fig)
	img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	gray = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	return cv2.resize(gray, output_shape)

if __name__ == '__main__':

	x = loadmat('data/ook_10p_0cm.mat')['data_10p_0cm'].T
	plt.imshow(visualization_block(x[1]))
	plt.show()


