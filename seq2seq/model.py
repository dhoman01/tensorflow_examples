# Copyright 2016 Dustin E. Homan. All Rights Reserved.
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
"""A seq2seq model to translate images of math equations to LaTeX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

class seq2seqModel(object):
    def __init__(self, config):
        self.config = config

        self.initializer = tf.random_uniform_initializer(
            minval=-0.08,
            maxval=0.08)

        self.x = tf.placeholder(tf.float32, [None, config.image_width * config.image_height * config.image_channels])
        self.y = tf.placeholder(tf.float32, [config.batch_size, 500])

        image = tf.split(1, config.image_height, self.x)
        formula = [self.y]

        with tf.variable_scope("seq2seq") as scope:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.rnn_size)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                            lstm_cell,
                            input_keep_prob=0.7,
                            output_keep_prob=0.7)
            if self.config.num_layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

            self.outputs, self.state = tf.nn.seq2seq.basic_rnn_seq2seq(image, formula, lstm_cell, scope=scope)

        batch_loss = self.calc_loss(self.outputs, formula)

        tf.contrib.losses.add_loss(batch_loss)
        total_loss = tf.contrib.losses.get_total_loss()

        # Add summeries
        tf.scalar_summary("batch_loss", batch_loss)
        tf.scalar_summary("total_loss", total_loss)
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        self.loss = batch_loss
        self.total_loss = total_loss

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def calc_loss(self, y_pred, y_true):
        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib. layers.fully_connected(
                inputs=y_pred,
                num_outputs=500,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope)
        targets = tf.unpack(y_true, axis=1)
        weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]
        l = tf.nn.seq2seq.sequence_loss(logits, targets, weights)
        return l

    def accuracy(self, y_pred, y_true):
        pred = tf.to_int32(tf.argmax(y_pred, 2))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y_true), tf.float32), name="pred_accuracy")
        return accuracy

    def step(self, sess, images, formulas):
        return sess.run([self.train_op, self.loss], feed_dict={self.x: images, self.y: formulas});
