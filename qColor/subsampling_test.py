import numpy as np
import itertools
import matplotlib.image as image
import matplotlib.pyplot as plt

from PIL import Image
from copy import deepcopy

from qColor.color_conv import ycbcr2rgb, rgb2ycbcr


img = Image.open('images/cat.jpeg')
img = img.resize((116, 116), Image.ANTIALIAS)
img = image.pil_to_array(img)[:,:,:3]

plt.imshow(img)
plt.show()

img = rgb2ycbcr(img)

def subsampling(img):
	new_img = deepcopy(img)

	it = np.nditer(img, flags=['multi_index'])
	indices_list = np.array([it.multi_index for _ in it])
	indices_list = indices_list.reshape(int((img.shape[0]*img.shape[0])/2), 2, 3, 3)
	for pair in indices_list:
		old_left = (img[pair[0][0][0]][pair[0][0][1]][pair[0][0][2]], img[pair[0][1][0]][pair[0][1][1]][pair[0][1][2]], img[pair[0][2][0]][pair[0][2][1]][pair[0][2][2]])
		old_right = (img[pair[1][0][0]][pair[1][0][1]][pair[1][0][2]], img[pair[1][1][0]][pair[1][1][1]][pair[1][1][2]], img[pair[1][2][0]][pair[1][2][1]][pair[1][2][2]])
		new_left, new_right = conv_pixels(old_left, old_right)

		new_img[pair[0][0][0]][pair[0][0][1]][pair[0][0][2]] = new_left[0]
		new_img[pair[0][1][0]][pair[0][1][1]][pair[0][1][2]] = new_left[1]
		new_img[pair[0][2][0]][pair[0][2][1]][pair[0][2][2]] = new_left[2]
		new_img[pair[1][2][0]][pair[1][2][1]][pair[1][2][2]] = new_right[0]
		new_img[pair[1][1][0]][pair[1][1][1]][pair[1][1][2]] = new_right[1]
		new_img[pair[1][2][0]][pair[1][2][1]][pair[1][2][2]] = new_right[2]

	return new_img

def subsampling_decoding(img):
	new_img = deepcopy(img)

	it = np.nditer(img, flags=['multi_index'])
	indices_list = np.array([it.multi_index for _ in it])
	indices_list = indices_list.reshape(int((img.shape[0]*img.shape[0])/2), 2, 3, 3)
	for pair in indices_list:
		old_left = (img[pair[0][0][0]][pair[0][0][1]][pair[0][0][2]], img[pair[0][1][0]][pair[0][1][1]][pair[0][1][2]], img[pair[0][2][0]][pair[0][2][1]][pair[0][2][2]])
		old_right = (img[pair[1][0][0]][pair[1][0][1]][pair[1][0][2]], img[pair[1][1][0]][pair[1][1][1]][pair[1][1][2]], img[pair[1][2][0]][pair[1][2][1]][pair[1][2][2]])
		new_right = conv_pixels_decoding(old_left, old_right)

		new_img[pair[1][2][0]][pair[1][2][1]][pair[1][2][2]] = new_right[0]
		new_img[pair[1][1][0]][pair[1][1][1]][pair[1][1][2]] = new_right[1]
		new_img[pair[1][2][0]][pair[1][2][1]][pair[1][2][2]] = new_right[2]

	return new_img

def conv_pixels(left: tuple, right:tuple):
	new_left = (left[0], int((int(left[1]) + int(right[1]))/2), int((int(left[2]) + int(right[2]))/2))
	new_right = (right[0], 128, 128)
	return new_left, new_right

def conv_pixels_decoding(left: tuple, right: tuple):
	new_right = (right[0], left[1], left[2])
	return new_right

new_img = subsampling(img)
print(new_img)
new_img = ycbcr2rgb(new_img)
img = ycbcr2rgb(img)

plt.imshow(img)
plt.show()

plt.imshow(new_img)
print(new_img)
plt.show()

new_img = rgb2ycbcr(new_img)
new_img = subsampling_decoding(new_img)
new_img = ycbcr2rgb(new_img)

plt.imshow(new_img)
plt.show()
