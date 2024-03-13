# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf
from models.common import retargeted_vgg


class ContentStyleModel(tf.keras.models.Model):
	def __init__(self, content_layers, style_layers):
		super().__init__()

