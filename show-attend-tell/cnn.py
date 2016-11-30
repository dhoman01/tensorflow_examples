# Copyright 2016 Dustin Homan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class CNN(object):
    def __init__(self, args):
        # Helper functions
        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)

        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)

        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[2, 2, 2, 2], padding='SAME')

        def max_pool_4x4(x):
          return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                                strides=[1, 4, 4, 1], padding='SAME')

        self.x = tf.placeholder(tf.float32, shape=[args.batch_size, args.image_width * args.image_height])

        with tf.variable_scope('input_cnn'):
            W_conv1 = weight_variable([32, 32, 1, 32])
            b_conv1 = bias_variable([32])

            x_image = tf.reshape(self.x, [-1, args.image_width, args.image_height, 1])

            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_4x4(h_conv1)

            # Add summaries
            tf.scalar_summary("weights", W_conv1)
            tf.scalar_summary("biases", b_conv1)
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)

        with tf.variable_scope('hidden_cnn1'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_4x4(h_conv2)

            # Add summaries
            tf.scalar_summary("weights", W_conv2)
            tf.scalar_summary("biases", b_conv2)
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)

        with tf.variable_scope('hidden_cnn2'):
            W_conv3 = weight_variable([1, 1, 64, 64])
            b_conv3 = bias_variable([64])

            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_4x4(h_conv3)

            self.keep_prob = tf.placeholder(tf.float32)
            h_conv3_drop = tf.reshape(tf.nn.dropout(h_pool3, self.keep_prob), [126, 64])

            W_conv3 = tf.reshape(W_conv3, [64, 64])

            self.output = tf.matmul(h_conv3_drop, W_conv3) + b_conv3

            # Add summaries
            tf.scalar_summary("weights", W_conv3)
            tf.scalar_summary("biases", b_conv3)
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)
