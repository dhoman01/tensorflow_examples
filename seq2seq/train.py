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
"""Train the model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time
from datetime import datetime

import tensorflow as tf

from model import seq2seqModel
from ops import DataLoader, Configuration

tf.app.flags.DEFINE_string("train_dir", "train_dir", "Directory to save checkpoints")
tf.app.flags.DEFINE_integer("num_steps", 1000000, "Number of epochs")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "Steps per checkpoint")

FLAGS = tf.app.flags.FLAGS

def main(_):
    config = Configuration()
    data_loader = DataLoader(config)
    model = seq2seqModel(config)

    with tf.Session() as sess:
        step_time, loss = 0.0, 0.0
        previous_losses = []
        tf.logging.info("%s: Starting training with %d steps" % (datatime.now(), FLAGS.num_steps))
        for current_step in range(FLAGS.num_steps):
            batch_x, batch_y = data_loader.next_batch()

            start_time = time.time()
            _, step_loss = model.step(sess, batch_x, batch_y)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint

            loss += step_loss / FLAGS.steps_per_checkpoint

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_fn)
                previous_losses.append(loss)

                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

if __name__ == "__main__":
    tf.app.run()
