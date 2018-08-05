import numpy as np
import cv2

def add_pos(arr):
	s = arr.shape
	result = np.empty((s[0], s[1] + 2, s[2], s[3]), dtype=np.float32)
	result[:,:s[1],:,:] = arr
	x = np.repeat(np.expand_dims(np.arange(s[3]) / float(s[3]), axis=0), s[2], axis=0)
	y = np.repeat(np.expand_dims(np.arange(s[2]) / float(s[2]), axis=0), s[3], axis=0)
	result[:,s[1] + 0,:,:] = x
	result[:,s[1] + 1,:,:] = np.transpose(y)
	return result

def auto_canny(image, sigma=0.0):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	grayed = np.where(gray < 20, 255, 0)

	lower = sigma*128 + 128
	upper = 255
	edged = cv2.Canny(image, lower, upper)

	return np.maximum(edged, grayed)

def save_image(x, fname):
	img = np.transpose(x * 255, (1, 2, 0))
	img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
	cv2.imwrite(fname, img)
