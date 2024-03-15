# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import tensorflow as tf
from models.common import retargeted_vgg


class DeepDream(tf.Module):
	def __init__(self, target_layers, loss_fn):
		super().__init__()
		self.model = retargeted_vgg(target_layers)
		self.loss_fn = loss_fn

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


