# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import matplotlib.pyplot as plt
import tensorflow as tf
from utilities import *
from func.train import train_loop
from loss_fn.loss_funcs import content_only_loss, style_only_loss, content_style_loss, dream_loss
from models.feature_extraction import FeatureExtractionModel
from models.content_style import ContentStyleModel
from models.deepdream import DeepDream
import tensorflow_hub as hub
tf.random.set_seed(231)

content_image = load_img('assets/content_image.jpeg') # load_img('assets/MQ_fountain.jpg')
style_image = load_img('assets/starry_night.jpg')
dream_image = Image.open('assets/Ameca2.jpg') # Image.open('assets/sky1024px.jpg')
max_dim = 500
dream_image.thumbnail((max_dim, max_dim))
dream_image = np.array(dream_image)
# style_image = load_img('assets/Style-Vassily_Kandinsky.jpeg')
# dream_image = load_img('assets/sky1024px.jpg', max_dim=500)# load_img('assets/Ameca2.jpg')


def run_fast_style_transfer():
	# fast style transfer using TF-Hub
	hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
	stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
	plot_tensor_as_image(stylized_image, save_as='outputs/stylized.png')


def get_optimizer():
	return tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
	# return tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(
	# 	initial_learning_rate=8.0, decay_steps=445,decay_rate=0.98
	# ))


def run_content_reconstruction(content_layers, epochs=10, steps_per_epoch=100):
	opt = get_optimizer()
	# opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
	image = tf.Variable(tf.random.uniform(content_image.shape, dtype=tf.dtypes.float32))
	model = FeatureExtractionModel(content_layers)
	target_content_features = model(content_image)
	train_loop(model, image, content_only_loss, opt, epochs, steps_per_epoch, target_content=target_content_features)
	# show and save image
	plot_tensor_as_image(image, save_as=f'outputs/content_reconstruction2/{content_layers[0]}.png')


# @tf.function()
def run_style_reconstruction(style_layers, epochs=10, steps_per_epoch=100):
	# opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
	opt = get_optimizer()
	image = tf.Variable(tf.random.uniform(content_image.shape, dtype=tf.dtypes.float32))
	model = FeatureExtractionModel(style_layers)
	target_style_features = model(style_image)
	train_loop(model, image, style_only_loss, opt, epochs, steps_per_epoch, target_style=target_style_features)
	# # show and save image
	plot_tensor_as_image(image, save_as=f'outputs/style_reconstruction2/{style_layers[-1]}.png')


def run_content_style_reconstruction(content_layers, style_layers, epochs=10, steps_per_epoch=100):
	opt = get_optimizer()
	# init from random
	# image = tf.Variable(tf.random.uniform(content_image.shape, dtype=tf.dtypes.float32))
	image = tf.Variable(content_image)
	model = ContentStyleModel(content_layers, style_layers)
	target_content_features = model(content_image)['content_outputs']
	target_style_features = model(style_image)['style_outputs']
	train_loop(model, image, content_style_loss, opt, epochs, steps_per_epoch, target_content=target_content_features, target_style=target_style_features)
	# # # # show and save image
	plot_tensor_as_image(image, save_as=f'outputs/content_style/{style_layers[-1]}.png')


def run_deepdream(layers, steps, step_size):
	dream_model = DeepDream(layers, dream_loss)
	image = tf.keras.applications.inception_v3.preprocess_input(dream_image)
	# image = tf.keras.applications.resnet50.preprocess_input(dream_image)
	# image = tf.keras.applications.vgg16.preprocess_input(dream_image)
	image = tf.convert_to_tensor(image)
	loss, image = dream_model(image, tf.constant(steps), tf.constant(step_size))
	image = deprocess(image)
	print('min value of image ', tf.reduce_min(image), 'max: ', tf.reduce_max(image))
	plot_array_as_image(np.array(image), save_as=f'outputs/deepdream/inception/dream_ameca_{layers[-1]}.png')


# step_size = tf.convert_to_tensor(step_size)
	# steps_remaining = steps
	# step = 0
	# while steps_remaining:
	# 	if steps_remaining > 100:
	# 		run_steps = tf.constant(100)
	# 	else:
	# 		run_steps = tf.constant(steps_remaining)
	# 	steps_remaining -= run_steps
	# 	step += run_steps
	#
	# 	loss, img = dream_model(dream_image, run_steps, tf.constant(step_size))
	#
	# 	print(f"Step {step}, loss {loss}")
	# result = deprocess(img)
	# display.clear_output(wait=True)
	# show(result)
	# return result




if __name__ == "__main__":
	# layers = ['mixed3', 'mixed5']
	# run_deepdream(layers, steps=1000, step_size=0.01)

	# run_fast_style_transfer()
	# print(tf.config.list_physical_devices('GPU'))
	style_layers = [['block1_conv1',], ['block1_conv1', 'block2_conv1'], ['block1_conv1', 'block2_conv1', 'block3_conv1'],
					['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1',],
					['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']]
	# # for layers in style_layers:
	run_style_reconstruction(style_layers[-1], epochs=10)

	# content_layers = [['block1_conv2'], ['block2_conv2'], ['block3_conv2'], ['block4_conv2'], ['block5_conv2']]
	# run_content_reconstruction(content_layers[4], epochs=10)
	# run_content_style_reconstruction(content_layers[-1], style_layers[-1], epochs=30)
#