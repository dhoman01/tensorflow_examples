import argparse
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class ArgumentParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_dir', type=str, default='data_dir',
                             help='directory containing the sub-dir "formula_images" and the files "im2latex_train.lst" and "im2latex_formulas.lst"')
        self.parser.add_argument('--train_dir', type=str, default='train_dir',
                             help='directory to save TF checkpoints')
        self.parser.add_argument('--rnn_size', type=int, default=128,
                         help='size of the RNN')
        self.parser.add_argument('--num_layers', type=int, default=128,
                            help='number of layers in the RNN')
        self.parser.add_argument('--batch_size', type=int, default=128,
                            help='minibatch size')
        self.parser.add_argument('--seq_length', type=int, default=28,
                            help='RNN sequence length')
        self.parser.add_argument('--num_classes', type=int, default=10,
                            help='number of classes')
        self.parser.add_argument('--num_steps', type=int, default=28,
                            help='number of timesteps')
        self.parser.add_argument('--num_epochs', type=int, default=20000,
                            help='number of epochs')
        self.parser.add_argument('--save_every', type=int, default=100,
                            help='save frequency')
        self.parser.add_argument('--grad_clip', type=float, default=5.,
                            help='clip gradients at this value')
        self.parser.add_argument('--learning_rate', type=float, default=0.002,
                            help='learning rate')
        self.parser.add_argument('--decay_rate', type=float, default=0.97,
                            help='decay rate for rmsprop')

class DataLoader(object):
    def __init__(self, args):
        self.batch_size = args.batch_size

    def next_batch(self):

        def getLabel(prediction):
            index = np.argmax(prediction)
            labels = ["0","1","2","3","4","5","6","7","8","9"]
            return labels[index]

        batch_x = []
        batch_y = []
        mnist_x, mnist_y = mnist.train.next_batch(self.batch_size * 3)
        for i in xrange(2, len(mnist_x), 3):
            x = np.concatenate([mnist_x[i - 2].reshape(28,28), mnist_x[i - 1].reshape(28,28), mnist_x[i].reshape(28,28)], axis=1)
            batch_x.append(x)
            y = np.array([getLabel(mnist_y[i - 2]), getLabel(mnist_y[i - 1]), getLabel(mnist_y[i])])
            batch_y.append(y)

        return batch_x, batch_y
