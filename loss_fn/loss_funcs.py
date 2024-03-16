# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf
from models.common import gram_matrix, gram_matrix_plain


# hyperparameters
content_weight = 1e4
style_weight = 1e-2


def content_only_loss(outputs: dict, **kwargs):
	target_content = kwargs.get('target_content')
	assert target_content is not None, 'should provide target content features!!!'
	loss = tf.add_n([tf.reduce_mean((outputs[name]-target_content[name])**2) for name in outputs.keys()])
	return loss / len(outputs)


def style_only_loss(outputs, **kwargs):
	target_style = kwargs.get('target_style')
	assert target_style is not None, 'should provide target style features!!!'
	loss = tf.add_n([
		tf.reduce_mean((gram_matrix(outputs[name])-gram_matrix(target_style[name]))**2) for name in outputs.keys()])
	return loss / len(outputs)


def content_style_loss(outputs, **kwargs):
	target_content = kwargs.get('target_content')
	target_style = kwargs.get('target_style')
	assert target_content is not None, 'should provide target content features!!!'
	assert target_style is not None, 'should provide target style_features!!!'
	content_loss = content_only_loss(outputs['content_outputs'], target_content=target_content)
	style_loss = style_only_loss(outputs['style_outputs'], target_style=target_style)
	return content_weight * content_loss + style_weight * style_loss


def dream_loss(image, feature_extraction_model):
	# image_batch = image[None, ...]  # or tf.expand_dims(image, axis=0)
	layer_activations = feature_extraction_model(image)
	if len(layer_activations) == 1:
		layer_activations = [layer_activations]
	losses = []
	for act in layer_activations:
		loss = tf.math.reduce_mean(act)
		losses.append(loss)
	return tf.reduce_mean(losses)

