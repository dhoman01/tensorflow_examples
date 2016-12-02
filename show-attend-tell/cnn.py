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

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

slim = tf.contrib.slim

class InceptionV3(object):
    def __init__(self, config, images, scope="InceptionV3"):
        tf.logging.info("InceptionV3 config: %s" % config)

        images = tf.reshape(images, [-1, config['image_height'], config['image_width'], config['image_channels']])

        tf.logging.info("Image reshaped to %s" % images.get_shape())
        if config['train_inception']:
            weights_regularizer = tf.contrib.layers.l2_regularizer(config['weight_decay'])
        else:
            weights_regularizer = None

        with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    trainable=config['train_inception']):
                with slim.arg_scope(
                        [slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=config['stddev']),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=None):
                    net, end_points = inception_v3_base(images, scope=scope)
                    with tf.variable_scope("logits"):
                        shape = net.get_shape()
                        net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                        net = slim.dropout(
                            net,
                            keep_prob=config['dropout_prob'],
                            is_training=config['train_inception'],
                            scope="dropout")
                        net = slim.flatten(net, scope="flatten")

        # Add summaries.
        if config['add_summaries']:
            for v in end_points.values():
                tf.contrib.layers.summaries.summarize_activation(v)

        self.net = net

def get_default_inception_configs(args):
    return {
        "train_inception": True,
        "dropout_prob": 0.8,
        "weight_decay": 0.00004,
        "stddev": 0.1,
        "add_summaries": True,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "image_channels": args.image_channels}

def setup_inception_initializer(mode, variables, ckpt_file):
    """Sets up the function to restore inception variables from checkpoint."""
    if mode != "inference":
        # Restore inception variables only.
        saver = tf.train.Saver(variables)

    def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
            ckpt_file)
        saver.restore(sess, ckpt_file)

    return restore_fn
