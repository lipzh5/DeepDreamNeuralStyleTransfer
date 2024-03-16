# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import tensorflow as tf
from models.common import retargeted_vgg, retargeted_resnet


class DeepDream(tf.Module):
	def __init__(self, target_layers, loss_fn):
		super().__init__()
		inception_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet') # retargeted_resnet(target_layers)
		outputs = [inception_model.get_layer(name).output for name in target_layers]
		print('inception model inputs: ', type(inception_model.inputs))
		self.model = tf.keras.Model(inputs=inception_model.inputs, outputs=outputs)
		# self.model = retargeted_vgg(target_layers, 'vgg16')
		self.loss_fn = loss_fn

	@tf.function(input_signature=(
		tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
		tf.TensorSpec(shape=[], dtype=tf.int32),
		tf.TensorSpec(shape=[], dtype=tf.float32)
	))
	# @tf.function()
	def __call__(self, image, steps, step_size):
		loss = tf.constant(0.0)
		for n in tf.range(steps):
			with tf.GradientTape() as tape:
				tape.watch(image)
				loss = self.loss_fn(image, self.model)
			gradients = tape.gradient(loss, image)
			gradients /= tf.math.reduce_std(gradients) + 1e-8
			image = image + gradients * step_size
			image = tf.clip_by_value(image, -1, 1)
		return loss, image


