# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf
from models.common import retargeted_vgg


class ContentStyleModel(tf.keras.models.Model):
	def __init__(self, content_layers, style_layers):
		super().__init__()
		self.vgg = retargeted_vgg(content_layers+style_layers)
		self.content_layers = content_layers
		self.style_layers = style_layers
		self.num_content_layers = len(content_layers)

	def call(self, inputs):
		"""Expects float inputs in [0, 1]"""
		inputs = inputs * 255.0
		preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
		outputs = self.vgg(preprocessed)
		content_outputs = outputs[:self.num_content_layers]
		style_outputs = outputs[self.num_content_layers:]
		return {'content_outputs': {name: val for name, val in zip(self.content_layers, content_outputs)},
				'style_outputs': {name: val for name, val in zip(self.style_layers, style_outputs)} }


