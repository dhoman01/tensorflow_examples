import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
        test_data = mnist.test.images[30]
        img = test_data.reshape((args.num_steps, args.seq_length))
        plt.imshow(img)
        plt.show()
        test_data = test_data.reshape((-1, args.num_steps, args.seq_length))
        test_label = mnist.test.labels[30]
        print("Correct label: %s" % getLabel(test_label))
        print("Prediction: %s" % getLabel(RNN.pred.eval(feed_dict={RNN.x: test_data})))

def getLabel(prediction):
    index = np.argmax(prediction)
    labels = ["0","1","2","3","4","5","6","7","8","9"]
    return labels[index]

if __name__ == '__main__':
    main()
