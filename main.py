# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import matplotlib.pyplot as plt
import tensorflow as tf
from utilities import *
from func.train import train_loop
from loss_fn.loss_funcs import content_only_loss
from models.content_reconstruction import ContentModel
import tensorflow_hub as hub

content_image = load_img('assets/MQ_fountain.jpg')
style_image = load_img('assets/starry_night.jpg')


# hyperparameters
style_weight = 1e-2
content_weight = 1e4

def run_fast_style_transfer():
	# fast style transfer using TF-Hub
	hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
	stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
	plot_tensor_as_image(stylized_image, save_as='stylized.png')
	# tensor_to_image(stylized_image)
	# img_show = stylized_image * 255
	# tensor = np.array(tensor, dtype=np.uint8)


def run_content_reconstruction(content_layers):
	opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
	image = tf.Variable(tf.random.uniform(content_image.shape, dtype=tf.dtypes.float32))
	model = ContentModel(content_layers)
	target_features = model(content_image)
	train_loop(model, image, target_features, content_only_loss, opt, epochs=10, steps_per_epoch=100)
	# show and save image
	image_show = image * 255
	image_show = np.array(image_show, dtype=np.uint8)
	plt.show(image_show)
	plt.savefig('content_recon.png')



if __name__ == "__main__":
	print(tf.__version__)
	content_layers = ['block5_conv2']
	run_content_reconstruction(content_layers)
