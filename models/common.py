# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf


def retargeted_vgg(content_layers):
	vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	vgg.trainable = False   # we update input images not the weights of vgg model
	outputs = [vgg.get_layer(name).output for name in content_layers]
	return tf.keras.Model([vgg.input], outputs)
