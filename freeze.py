from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from models import EmoNet
from tensorflow.python.framework import graph_util

FLAGS = None

def main(FLAGS):
	sess = tf.InteractiveSession()

	m = EmoNet()

	x_input = tf.placeholder(shape=[1, 48, 48, 1], dtype=tf.float32, name="image")
	#x = tf.reshape(x_input, [1, 48, 48, 1])

	logits = m.build_network(x_input, is_training=False)
	predictions = tf.nn.softmax(logits, axis=-1, name='softmax')

	saver = tf.train.Saver(max_to_keep=50)
	saver.restore(sess, FLAGS.start_checkpoint)

	frozen_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['softmax'])
	tf.train.write_graph(frozen_graph_def, os.path.dirname(FLAGS.output_file), os.path.basename(FLAGS.output_file), as_text=False)
	tf.logging.info('Saved frozen graph to %s', FLAGS.output_file)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--start_checkpoint",
		type=str,
		default=None,
		help="Model save dir")
	parser.add_argument(
		'--output_file', type=str, help='Where to save the frozen graph.')
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
