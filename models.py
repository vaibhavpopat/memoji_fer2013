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

		#print(vaibhavpopat)

		
