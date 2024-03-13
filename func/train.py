# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf
from utilities import *
import time


@tf.function()
def train_step(model, image_to_gen, target_features, loss_func, optimizer):
	with tf.GradientTape() as tape:
		outputs = model(image_to_gen)
		loss = loss_func(outputs, target_features)
	grad = tape.gradient(loss, image_to_gen)
	optimizer.apply_gradients([(grad, image_to_gen)])
	image_to_gen.assign(clip_0_1(image_to_gen))


def train_loop(model, image_to_gen, target_features, loss_func, optimizer, epochs=10, steps_per_epoch=100):
	start = time.time()
	step = 0
	for n in range(epochs):
		for m in range(steps_per_epoch):
			train_step(model, image_to_gen, target_features, loss_func, optimizer)
			step += 1
		print(f'Train step: {step}')

	print(f'Total time: {time.time() - start}')


