import argparse

class ArgumentParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_dir', type=str, default='data_dir',
                             help='directory containing the sub-dir "formula_images" and the files "im2latex_train.lst" and "im2latex_formulas.lst"')
        self.parser.add_argument('--train_dir', type=str, default='train_dir',
                             help='directory to save TF checkpoints')
        self.parser.add_argument('--num_layers', type=int, default=128,
                            help='number of layers in the RNN')
        self.parser.add_argument('--batch_size', type=int, default=500,
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
