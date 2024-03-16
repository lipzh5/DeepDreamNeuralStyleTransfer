# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf


def gram_matrix(input_tensor):
	result = tf.linalg.einsum('bijc, bijd->bcd', input_tensor, input_tensor)
	input_shape = tf.shape(input_tensor)
	num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
	return result/num_locations


def gram_matrix_plain(input_tensor):
	# remove batch dim
	input_tensor = input_tensor[0]  # (H, W, C) , C is the channel size or #filters
	vectorized_fea = tf.reshape(input_tensor, (tf.shape(input_tensor)[0], -1))
	return tf.matmul(vectorized_fea, tf.transpose(vectorized_fea))


def retargeted_vgg(target_layers, name='vgg19'):
	vgg_model = tf.keras.applications.VGG19 if name == 'vgg19' else tf.keras.applications.VGG16
	vgg = vgg_model(include_top=False, weights='imagenet')
	vgg.trainable = False   # we update input images not the weights of vgg model
	outputs = [vgg.get_layer(name).output for name in target_layers]
	return tf.keras.Model([vgg.input], outputs)


def retargeted_resnet(target_layers):
	resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
	resnet.trainable = False
	outputs = [resnet.get_layer(name).output for name in target_layers]
	return tf.keras.Model([resnet.input], outputs)



if __name__ == "__main__":
	resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
	print(resnet.summary())
	# from utilities import *
	# content_image = load_img('../assets/content_image.jpeg')
	# model = retargeted_vgg(['block1_conv1', 'block1_conv2'])
	# preprocessed = tf.keras.applications.vgg19.preprocess_input(content_image)
	# outputs = model(preprocessed)
	# print('outputs shape: ', type(outputs), type(outputs[0]), tf.shape(outputs[0]))
