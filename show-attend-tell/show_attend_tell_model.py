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

########################################################################
# Ideas and guidence from these wonderful people                       #
#    https://raw.githubusercontent.com/tensorflow/models/master/im2txt #
########################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from cnn import CNN
from ops import input_ops, image_processing

class Model(object):
    def __init__(self, args, mode):
        self.config = args
        self.mode = mode

        # Reader for the input data.
        self.reader = tf.WholeFileReader()

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        self.image_embeddings = tf.placeholder(tf.float32, [self.config.batch_size, self.config.embedding_size])

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Used to initialize variables in various networks
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

    def is_training(self):
        return self.mode == "training"

    def process_image(self, encoded_image, tread_id=0):
        return image_processing.process_image(encoded_image,
                    is_training=self.is_training(),
                    height=self.config.image_height,
                    width=self.config.image_width,
                    thread_id=thread_id,
                    channels=self.config.image_channels,
                    image_format=self.config.image_format)

    def build_inputs(self):
        if self.mode == "inference":
            # Input is fed via placeholders
            image_feed = tf.placeholder(tf.string, shape=[], name="image_feed")
            input_feed = tf.placeholder(tf.int64, shape=[None], name="input_feed")

            # Insert batch dimensions
            images = tf.expand_dims(self.process_image(image_feed), 0)
            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_mask = None
        else:
            # Prefetch serialized SequenceExample protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads)

            # Image processing and random distortion. Split across multiple threads
            # with each thread applying a slightly different distortion.
            assert self.config.num_preprocess_threads % 2 == 0
            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                encoded_image, caption = input_ops.parse_sequence_example(
                    serialized_sequence_example,
                    image_feature=self.config.image_feature_name,
                    caption_feature=self.config.caption_feature_name)
                image = self.process_image(encoded_image, thread_id=thread_id)
                images_and_captions.append([image, caption])

            # Batch inputs.
            queue_capacity = (2 * self.config.num_preprocess_threads *
                self.config.batch_size)
            images, input_seqs, target_seqs, input_mask = (
                input_ops.batch_with_dynamic_pad(images_and_captions,
                    batch_size=self.config.batch_size,
                    queue_capacity=queue_capacity))

        self.images = images
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask

    def build_image_embeddings(self):
        cnn = CNN(self.config)
        cnn.x = self.images
        cnn_outputs = cnn.outputs

        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=cnn_outputs,
                num_outputs=self.config.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

        tf.constant(self.config.embedding_size, name="embedding_size")

        self.image_embeddings = image_embeddings

    def build_seq_embeddings(self):
        with tf.variable_scope("seq_embedding"):
            embedding_map = tf.get_variable("map", shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings

    def build_model(self):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.rnn_size)
        if self.config.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.config.num_layers)

        def _decoder_fn(decoder_inputs, initial_state, cell, num_symbols, embedding_size, scope, initial_state_attention=False):
            top_states = [array_ops.reshape(e, [-1, 1, cell.output_size]) for e in self.image_embeddings]
            attention_states = array_ops.concat(1, top_states)
            return tf.nn.seq2seq.embedding_attention_decoder(decoder_inputs=decoder_inputs,
                                                             initial_state=initial_state,
                                                             attention_states=attention_states,
                                                             cell=cell,
                                                             num_symbols=num_symbols,
                                                             embedding_size=embedding_size,
                                                             scope=scope,
                                                             initial_state_attention=initial_state_attention)

        with tf.variable_scope("attend") as attend_scope:
            zero_state = cell.zero_state(batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
            _, initial_state = cell(self.image_embeddings, zero_state)

            # allow attend variables to be reused.
            attend_scope.reuse_variables()

            if self.mode == "inference":
                tf.concat(1, initial_state, name="initial_state")

                state_feed = tf.placeholder(tf.float32, shape=[None, sum(cell.state_size)], name="state_feed")
                state_tuple = tf.split(1, 2, state_feed)

                # Run a single LSTM step
                outputs, state_tuple = cell(inputs=tf.squeeze(self.seq_embeddings, squeeze_dims=[1]), state=state_tuple)

                # Concatenate the state
                tf.concat(1, state_tuple, name="state")
            else:
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                outputs, state = _decoder_fn(
                    decoder_inputs=self.seq_embeddings,
                    initial_state=initial_state,
                    cell=cell,
                    num_symbols=sequence_length,
                    embedding_size=self.config.embedding_size,
                    scope=attend_scope)

        # Stace batches
        outputs = tf.reshape(outputs, [-1, cel.output_size])

        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib. layers.fully_connected(
                inputs=outputs,
                num_outputs=self.config.vocab_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope)

        if self.mode == "inference":
            tf.nn.softmax(logits, name="softmax")
        else:
            targets = tf.reshape(self.target_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

            # Compute losses
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
            batch_loss = tf.div(tf.reduce_sum(tf.mul(losses, weights)), tf.reduce_sum(weights), name="batch_loss")
            tf.contrib.losses.add_loss(batch_loss)
            total_loss = tf.contrib.losses.get_total_loss()

            # Add summeries
            tf.scalar_summary("batch_loss", batch_loss)
            tf.scalar_summary("total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses
            self.target_cross_entropy_loss_weights = weights

    def setup_global_step(self):
        global_step = tf.Variable(
            intial_value=0,
            name="global_step",
            trainable=False,
            collection=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_image_embeddings()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_global_step()
