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

from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os
import random
import sys
import threading
import argparse
import shutil
import nltk
import uuid

from PIL import Image

import tensorflow as tf

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data_dir',
    help='The directory to store the data in')
parser.add_argument('--train_dir', type=str, default='train',
    help='The directory in the data dir to store training images')
parser.add_argument('--test_dir', type=str, default='test',
    help='The directory in the data dir to store the testing images')
parser.add_argument('--num_of_train', type=int, default=50000,
    help='The number of training images to generate')
parser.add_argument('--num_of_test', type=int, default=9996,
    help='The number of testing images to generate. Needs to be <9996 and x % 3 == 0')
args = parser.parse_args()

if not os.path.isdir(args.data_dir):
    os.mkdir(args.data_dir)
    os.mkdir(os.path.join(args.data_dir, args.train_dir))
    os.mkdir(os.path.join(args.data_dir, args.test_dir))
if not os.path.isdir(os.path.join(args.data_dir, args.train_dir)):
    os.mkdir(os.path.join(args.data_dir, args.train_dir))
if not os.path.isdir(os.path.join(args.data_dir, args.test_dir)):
    os.mkdir(os.path.join(args.data_dir, args.test_dir))

try:
    shutil.rmtree(os.path.join(args.data_dir, args.train_dir, '*'))
except:
    print('Train dir is already empty')

try:
    shutil.rmtree(os.path.join(args.data_dir, args.test_dir, '*'))
except:
    print('Test dir is already empty')

labels = ["0","1","2","3","4","5","6","7","8","9"]
def onehot2label(caption):
    index = np.argmax(caption)
    return labels[index]

train_images = []
train_labels = []

for i in range(args.num_of_train):
    images, captions = mnist.train.next_batch(3)

    image = np.concatenate((images[0].reshape(28,28), images[1].reshape(28,28), images[2].reshape(28,28)), axis=1)
    label = onehot2label(captions[0]) + " " + onehot2label(captions[1]) + " " + onehot2label(captions[2])
    image = Image.fromarray(np.uint8(image*255))
    filename = os.path.join(args.data_dir, args.train_dir, "%s.png" % uuid.uuid4())
    image.save(filename, 'PNG')
    train_images.append(filename)
    train_labels.append(label)

images = mnist.test.images
captions = mnist.test.labels

test_images = []
test_labels = []

for i in range(args.num_of_test):
    image = np.concatenate((images[i].reshape(28,28), images[i + 1].reshape(28,28), images[i + 2].reshape(28,28)), axis=1)
    label = onehot2label(captions[i]) + " " + onehot2label(captions[i + 1]) + " " + onehot2label(captions[i + 2])
    image = Image.fromarray(np.uint8(image * 255))
    filename = os.path.join(args.data_dir, args.test_dir, "%s.png" % uuid.uuid4())
    image.save(filename, 'PNG')
    test_images.append(filename)
    test_labels.append(label)

tf.flags.DEFINE_string("output_dir", "/data_dir/", "Output data directory.")

tf.flags.DEFINE_string("start_word", "<GO>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</EOS>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 1,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "data_dir/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.
        Args:
        vocab: A dictionary of word to word_id.
        unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id
class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_png = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._encoded_png, channels=3)

  def decode_png(self, encoded_png):
    image = self._sess.run(self._decode_png,
                           feed_dict={self._encoded_png: encoded_png})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _create_vocab(captions):
  """Creates the vocabulary of word to word_id.
  The vocabulary is saved to disk in a text file of word counts. The id of each
  word in the file is its corresponding 0-based line number.
  Args:
    captions: A list of lists of strings.
  Returns:
    A Vocabulary object.
  """
  print("Creating vocabulary.")
  counter = Counter()
  for c in captions:
    counter.update(c)
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

  # Create the vocabulary dictionary.
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab

def _process_caption(caption):
    """Processes a caption string into a list of tonenized words.
    Args:
    caption: A string caption.
    Returns:
    A list of strings; the tokenized caption.
    """
    tokenized_caption = [FLAGS.start_word]
    tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
    tokenized_caption.append(FLAGS.end_word)
    return tokenized_caption

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _to_sequence_example(filename, caption, decoder, vocab):
  """Builds a SequenceExample proto for an image-caption pair.
  Args:
    image: An ImageMetadata object.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.
  Returns:
    A SequenceExample proto.
  """
  with tf.gfile.FastGFile(filename, "r") as f:
    encoded_image = f.read()

  try:
    decoder.decode_png(encoded_image)
  except (tf.errors.InvalidArgumentError, AssertionError):
    print("Skipping file with invalid PNG data: %s" % filename)
    return

  context = tf.train.Features(feature={
      "image/data": _bytes_feature(encoded_image),
  })

  caption_ids = [vocab.word_to_id(word) for word in caption]
  feature_lists = tf.train.FeatureLists(feature_list={
      "image/caption": _bytes_feature_list(caption),
      "image/caption_ids": _int64_feature_list(caption_ids)
  })
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example

# images and labels array as input
def convert_to(images, labels, name, dir, vocab):
  filename = os.path.join(dir, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  decoder = ImageDecoder()
  for index in range(len(labels)):
    caption = _process_caption(labels[index])
    example = _to_sequence_example(images[index], caption, decoder, vocab)
    writer.write(example.SerializeToString())

vocab = _create_vocab(train_labels + test_labels)

convert_to(train_images, train_labels, "training", os.path.join(args.data_dir, args.train_dir), vocab)
convert_to(test_images, test_labels, "testing", os.path.join(args.data_dir, args.test_dir), vocab)
