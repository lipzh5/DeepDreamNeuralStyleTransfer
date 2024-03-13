# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf
from models.common import retargeted_vgg


class FeatureExtractionModel(tf.keras.models.Model):
	"""Used to extract feature maps for target layers"""
	def __init__(self, target_layers):
		super().__init__()
		self.vgg = retargeted_vgg(target_layers)

	def call(self, inputs):
		"""Expects float inputs in [0, 1]"""
		inputs = inputs * 255.0
		preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
		return self.vgg(preprocessed)




if __name__ == "__main__":
	print(type(tf.keras))