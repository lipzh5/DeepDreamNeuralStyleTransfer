# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf
from models.common import gram_matrix, gram_matrix_plain


def content_only_loss(outputs: dict, target_features: dict):
	return tf.add_n([tf.reduce_mean((outputs[name]-target_features[name])**2) for name in outputs.keys()])


def style_only_loss(outputs, target_features):
	return tf.add_n([
		tf.reduce_mean((gram_matrix(outputs[name])-gram_matrix(target_features[name]))**2) for name in outputs.keys()])


