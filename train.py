from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import matplotlib.image as mig

from models import EmoNet

FLAGS = None

def loadData(file_name, batch_size):
	x = np.load(file_name+"_x.npy")
	y = np.load(file_name+"_y.npy")
	print(x.shape)
	
	#f = open(file_name+"_ydata", "w")
	#for yy in y:
	#	f.write(str(yy))
	#	f.write("\n")
	#f.close()
	#img_g = np.ones([48, 48, 3])
	
	#print(y)
	count = len(x) // batch_size
	if len(x) % batch_size != 0:
		count += 1
	x_batch = np.empty([count, batch_size, x.shape[1], x.shape[2], x.shape[3]])
	y_batch = np.empty([count, batch_size], dtype=np.int32)
	for i in range(count-1):
		x_batch[i] = x[i*batch_size:(i+1)*batch_size] / 255
		y_batch[i] = y[i*batch_size:(i+1)*batch_size]
		#mig.imsave(file_name+str(i)+"_x.png", img_g*x[i*batch_size]/255, vmax=1.0, vmin=0.0)
	x_batch[-1] = x[-batch_size:] / 255
	y_batch[-1] = y[-batch_size:]

	return (x_batch, y_batch, count)

def main(FLAGS):
	sess = tf.InteractiveSession()

	m = EmoNet()

	x_input = tf.placeholder(shape=[FLAGS.batch_size, 48, 48, 1], dtype=tf.float32)
	y_input = tf.placeholder(shape=[FLAGS.batch_size], dtype=tf.int64)

	logits, dropout_placeholder = m.build_network(x_input, is_training=True)
	predictions = tf.argmax(logits, axis=-1)

	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=logits))
	accuracy = tf.reduce_mean(tf.cast(tf.equal(y_input, predictions), tf.float32))

	optimizer = tf.train.AdamOptimizer(FLAGS.learn_rate)
	gradsvars = optimizer.compute_gradients(cost, var_list=tf.trainable_variables())
	grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 50)
	optimizer_fn = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))

	cel = tf.summary.scalar('cross-entropy-loss', cost)
	acc = tf.summary.scalar('accuracy', accuracy)
	summary = tf.summary.merge([cel, acc])

	train_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, "train"))
	test_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, "test"))

	global_step = tf.Variable(tf.constant(0, dtype=tf.int32), trainable=False)
	increment_global_step = tf.assign(global_step, global_step + 1)

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(max_to_keep=50)

	if FLAGS.start_checkpoint: #ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Imported")
		tf.logging.info("Imported model")
		saver.restore(sess, FLAGS.start_checkpoint)

	tf.train.write_graph(sess.graph_def, FLAGS.model_dir, 'graph.pbtxt')

	x_train, y_train, train_epochs = loadData(os.path.join(FLAGS.data_dir, "train"), FLAGS.batch_size)
	x_test, y_test, test_epochs = loadData(os.path.join(FLAGS.data_dir, "test"), FLAGS.batch_size)

	for i in range(FLAGS.epochs):
		for j in range(train_epochs):
			loss, _, s = sess.run([cost, optimizer_fn, summary], feed_dict={x_input: x_train[j], y_input: y_train[j], dropout_placeholder: 0.5}) #m.runBatch(sess, train_writer, train_batches_x[j], train_batches_y[j], sess.run(global_step))
			train_writer.add_summary(s, sess.run(global_step))
			sess.run(increment_global_step)
			print(loss) #for slow computer

		saver.save(sess, os.path.join(FLAGS.model_dir, "EmoNet"), global_step=global_step)

		test_acc = 0
		print("Test:")
		for k in range(test_epochs):
			a, _, s = sess.run([accuracy, optimizer_fn, summary], feed_dict={x_input: x_test[k], y_input: y_test[k], dropout_placeholder: 1.0}) #m.runBatch(sess, train_writer, train_batches_x[j], train_batches_y[j], sess.run(global_step))
			test_writer.add_summary(s, sess.run(global_step))
			test_acc += a

		print(test_acc/test_epochs)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir",
		type=str,
		default="data/",
		help="Data dir")
	parser.add_argument("--model_dir",
		type=str,
		default="data/",
		help="Model save dir")
	parser.add_argument("--epochs",
		type=int,
		default=5,
		help="Model save dir")
	parser.add_argument("--start_checkpoint",
		type=str,
		default=None,
		help="Model save dir")
	parser.add_argument("--learn_rate",
		type=float,
		default=0.01,
		help="Learn rate")
	parser.add_argument("--batch_size",
		type=int,
		default=256,
		help="Size of batch")
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
