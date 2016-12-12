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
"""Helper operations for training"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

class Configuration(object):
    def __init__(self):
        self.num_layers = 3
        self.rnn_size = 128

        self.learning_rate = 0.2

        self.image_width = 750
        self.image_height = 750
        self.image_channels = 3

        self.batch_size = 50

        self.input_file_pattern = "/home/dustin/im2latex/data_dir/*.tfrecords"


class DataLoader(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.config  = config
        config.input_file_pattern

        self.filename_q = tf.train.string_input_producer(tf.gfile.Glob(config.input_file_pattern), shuffle=True, capacity=16, name="shard_queue")
        self.reader = tf.WholeFileReader()

    def _parse_sequence_example(self, serialized):
        context, sequence = tf.parse_single_sequence_example(
            encoded_image,
            context_features={
                'image/data': tf.FixedLenFeature([], dtype=tf.string)
            },
            sequence_features={
                'image/formula_ids': tf.FixedLenFeature([], dtype=tf.int64)
            })

        encoded_image = context['image/data']
        formula = sequence['image/formula_ids']
        return encoded_image, formula

    def _process_image(self, encoded_image):
        def image_summary(name, image):
            tf.image_summary(name, tf.expand_dims(image, 0))

        with tf.name_scope("decode", values=[encoded_image]):
            image = tf.decode_png(encoded_image, channels=3)

        image = tf.image.convert_image_dtype(image, dtype=tf.int32)
        image_summary("original_image", image)

        image = tf.image.resize_image(image, size=[self.config.image_height, self.config.image_width], method=tf.image.ResizeMethod.BILINEAR)

        image_summary("resize_image", image)

        image = tf.reshape(image, [-1])

        print('image ' % image)
        return image

    def next_batch(self):
        batch_x = []
        batch_y = []

        for i in range(self.batch_size):
            serialized = self.reader.read(self.filename_q.dequeue())
            encoded_image, formula = self._parse_sequence_example(serialized)
            image = self._process_image(encoded_image)
            batch_x.append(image)
            batch_y.append(formula)

        return batch_x, batch_y
