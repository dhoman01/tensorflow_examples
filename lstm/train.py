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
    RNN = Model(args)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.all_variables())
        checkpoint_path = os.path.join(args.train_dir, 'model.ckpt')

        step = 1
        print("Training started...")
        while step < args.num_epochs:
            # Get and reshape batch
            batch_x, batch_y = mnist.train.next_batch(args.batch_size)
            batch_x = batch_x.reshape((args.batch_size, args.num_steps, args.seq_length))

            # Run optimizer
            sess.run(RNN.optimizer, feed_dict={RNN.x: batch_x, RNN.y: batch_y})
            if step % args.save_every == 0:
                # Calculate accuracy
                accuracy = sess.run(RNN.accuracy, feed_dict={RNN.x: batch_x, RNN.y: batch_y})
                # Calculate loss
                loss = sess.run(RNN.cost, feed_dict={RNN.x: batch_x, RNN.y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(accuracy))
                saver.save(sess, checkpoint_path, global_step=step)

            step += 1
        saver.save(sess, checkpoint_path)
        print("Training finished...")

if __name__ == '__main__':
    main()
