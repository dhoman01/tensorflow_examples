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
    CNN = Model(args)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')
        saver.restore(sess, checkpoint_path)
        test_data = mnist.test.images[0]
        img = test_data.reshape((args.num_steps, args.seq_length))
        plt.imshow(img)
        plt.show()
        test_label = mnist.test.labels[0]
        print("Correct label: %s" % getLabel(test_label))
        pred = tf.nn.softmax(CNN.y_conv)
        print("Prediction: %s" % getLabel(pred.eval(feed_dict={CNN.x: [test_data], CNN.keep_prob: 1.0})))

def getLabel(prediction):
    index = np.argmax(prediction)
    labels = ["0","1","2","3","4","5","6","7","8","9"]
    return labels[index]

if __name__ == '__main__':
    main()
