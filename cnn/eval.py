import os
import tensorflow as tf

from model import Model
from utils import ArgumentParser

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def main():
    args = ArgumentParser().parser.parse_args()
    eval(args)

def eval(args):
    CNN = Model(args)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')
        saver.restore(sess, checkpoint_path)
        test_data = mnist.test.images
        test_labels = mnist.test.labels
        print("\ntest accuracy %g" % CNN.accuracy.eval(feed_dict={CNN.x: test_data, CNN.y: test_labels}))

if __name__ == '__main__':
    main()
