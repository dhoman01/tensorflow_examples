import os
import tensorflow as tf

from utils import ArgumentParser
from model import Model

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def main():
    args = ArgumentParser().parser.parse_args()

    # Make needed dirs if not found
    if os.path.isdir(os.path.join(os.getcwd(),args.data_dir)) is not True:
        os.makedirs(args.data_dir)
    if os.path.isdir(os.path.join(os.getcwd(), args.train_dir)) is not True:
        os.makedirs(args.train_dir)

    # Begin training
    train(args)

def train(args):
    CNN = Model(args)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')

        print "Training started..."
        for step in range(args.num_epochs):
            batch = mnist.train.next_batch(args.batch_size)
            if step % args.save_every == 0:
                train_accuracy = CNN.accuracy.eval(feed_dict={
                    CNN.x:batch[0], CNN.y: batch[1], CNN.keep_prob: 1.0})
                print("step %d, training accuracy %g"%(step, train_accuracy))
                saver.save(sess, checkpoint_path, global_step=step)
            CNN.train_step.run(feed_dict={CNN.x: batch[0], CNN.y: batch[1], CNN.keep_prob: 0.5})
        saver.save(sess,checkpoint_path)
        print "Training complete..."


if __name__ == '__main__':
    main()
