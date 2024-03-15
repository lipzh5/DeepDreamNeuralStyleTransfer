# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def plot_tensor_as_image(tensor, save_as=''):
	tensor = tensor * 255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor) > 3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
	plt.figure(layout='tight')
	plt.axis('off')
	plt.imshow(tensor)
	if save_as:
		plt.savefig(save_as)


def tensor_to_image(tensor):
	tensor = tensor * 255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor) > 3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
	return Image.fromarray(tensor)# run_fast_style_transfer()
	# print(tf.config.list_physical_devices('GPU'))


def load_img(path_to_img):
	max_dim = 512
	img = tf.io.read_file(path_to_img)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	shape = tf.cast(tf.shape(img)[:-1], tf.float32)
	long_dim = max(shape)
	scale = max_dim / long_dim

	new_shape = tf.cast(shape * scale, tf.int32)

	img = tf.image.resize(img, new_shape)
	img = img[tf.newaxis, :]
	return img


def imshow(image, title=None):
	if len(image.shape) > 3:
		image = tf.squeeze(image, axis=0)
	plt.imshow(image)
	if title:
		plt.title(title)


def clip_0_1(image):
	return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def plot_images_in_a_row(path, save_as):
	for dirpath, dirname, filenames in os.walk(path):
		num_figs = len(filenames)
		plt.figure(figsize=(18, 18), layout='tight')
		filenames.sort()
		print('filenames sorted: ', filenames)
		for i, name in enumerate(filenames):
			if not name.endswith('.png'):
				continue
			plt.subplot(1, num_figs, i+1)
			plt.imshow(np.array(Image.open(os.path.join(dirpath, name))))
			plt.axis('off')
			# last_name = name.split('.')[0] # block2_conv1
			# plt.title(name.split('.')[0])
		plt.savefig(os.path.join(path, save_as))
		# plt.show()



if __name__ == "__main__":
	plot_images_in_a_row('outputs/style_reconstruction2', save_as='style_reconstructions.png')
