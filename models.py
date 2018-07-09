from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class EmoNet():
	def __init__(self, input_size=48, n_outputs=6, num_blocks=4, num_layers_per_block=2):
		self.input_size = input_size
		self.n_outputs = n_outputs
		self.num_blocks = num_blocks
		self.num_layers_per_block = num_layers_per_block

	def residual_dilated_block(self, x, filters, kernel_size, dilation, name):
		with tf.variable_scope(name) as scope:
			filter = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, dilation_rate=dilation, padding='same', activation=tf.nn.tanh, name="filter")
			gate = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, dilation_rate=dilation, padding='same', activation=tf.nn.sigmoid, name="gate")

			out = filter*gate
			out = tf.layers.conv2d(out, filters=filters, kernel_size=1, padding='same', activation=tf.nn.tanh, name="out")

		return out + x

	def build_network(self, input_placeholder, is_training=False):
		#input_placeholder (b, input_size, input_size, 1)

		if is_training:
			dropout_placeholder = tf.placeholder(shape=[], dtype=tf.float32)

		n_filters = 16
		last_size = self.input_size
		x = tf.layers.conv2d(input_placeholder, filters=n_filters, kernel_size=3, padding='same', activation=tf.nn.tanh, name="head")
		#x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
		#last_size = last_size // 2

		for b in range(self.num_blocks):
			for l in range(self.num_layers_per_block):
				name = "-".join(["rdb", str(b), str(l)])
				x = self.residual_dilated_block(x, n_filters, 3, 1, name)
			x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
			n_filters *= 2
			x = tf.layers.conv2d(x, filters=n_filters, kernel_size=1, padding='same', activation=None, name="scale"+str(b))
			last_size = last_size // 2

		x = tf.reshape(x, [input_placeholder.get_shape()[0], n_filters*last_size*last_size])
		if is_training:
			x = tf.nn.dropout(x, dropout_placeholder)

		logits = tf.contrib.layers.fully_connected(x, self.n_outputs, activation_fn=None, scope="classification")

		if is_training:
			return logits, dropout_placeholder
		return logits

