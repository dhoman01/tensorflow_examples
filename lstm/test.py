import os
import tensorflow as tf
from PIL import Image

from utils import ArgumentParser
from model import Model

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def main():
    args = ArgumentParser().parser.parse_args()
    predict(args)

def predict(args):
    RNN = Model(args)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')
        saver.restore(sess, checkpoint_path)
        test_data = mnist.test.images[3]
        test_data = test_data.reshape((args.num_steps, args.seq_length))
        print test_data
        img = Image.fromarray(test_data, "F")
        img.show();
        test_labels = mnist.test.labels[3]
        print test_labels
        print("\n prediction %g" % RNN.pred.eval(feed_dict={RNN.x: test_data}))

if __name__ == '__main__':
    main()
