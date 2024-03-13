# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import tensorflow as tf


def content_only_loss(outputs, target_features):
	return tf.reduce_mean((outputs-target_features)**2)

