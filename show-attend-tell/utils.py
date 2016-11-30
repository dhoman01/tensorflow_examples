import argparse

class ArgumentParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--data_dir', type=str, default='data_dir',
            help='directory containing the training data')
        self.parser.add_argument('--train_dir', type=str, default='train_dir',
            help='directory to save TF checkpoints')
        self.parser.add_argument('--eval_dir', type=str, default='eval_dir',
            help='directory to save Eval summaries')
        self.parser.add_argument('--batch_size', type=int, default=126,
            help='the size of each training batch')
        self.parser.add_argument('--initializer_scale', type=float, default=0.08,
            help='Bounds for random variables (lower is negative of given value)')
        self.parser.add_argument('--embedding_size', type=int, default=512,
            help='LSTM input dimensionality')
        self.parser.add_argument('--num_lstm_units', type=int, default=512,
            help='LSTM output dimensionality')
        self.parser.add_argument('--lstm_droput_keep_prob', type=float, default=0.7,
            help='The dropout keep probability applied to LSTM variables')
        self.parser.add_argument('--vocab_size', type=int, default=20,
            help='Number of unique "words" in the vocab (+1 for <UNK>). Must be greater than or equal to actual vocab size.')
        self.parser.add_argument('--num_preprocess_threads', type=int, default=4,
            help='Number of threads used for preprocessing. Should be multiple of 2')
        self.parser.add_argument('--image_height', type=int, default=28,
            help='The height dimension of input images')
        self.parser.add_argument('--image_width', type=int, default=84,
            help='The width dimension of input images')
        self.parser.add_argument('--num_epochs', type=int, default=20000,
            help='The number of training epochs')
        self.parser.add_argument('--learning_rate', type=float, default=2.0,
            help='The learning rate of the model')
        self.parser.add_argument('--decay_factor', type=float, default=0.5,
            help='The decay factor of the learning rate')
        self.parser.add_argument('--num_epochs_per_decay', type=int, default=8,
            help='The number of epochs trained per decay cycle')
        self.parser.add_argument('--clip_gradients', type=float, default=5.0,
            help='The clip gradients')
        self.parser.add_argument('--max_ckpt_to_keep', type=int, default=5,
            help='The maximum number of checkpoints to save')
        self.parser.add_argument('--optimizer', type=str, default="Adam",
            help='The optimizer used for training the model')
        self.parser.add_argument('--log_every_n_steps', type=int, default=100,
            help='Frequency to print out log messages')
        self.parser.add_argument('--input_file_pattern', type=str, default="data_dir/train/*",
            help='The file pattern of the input images')
        self.parser.add_argument('--values_per_input_shard', type=int, default=2300,
            help='The ~number of values per input shard')
        self.parser.add_argument('--input_queue_capacity_factor', type=int, default=2,
            help="The min number of shards to keep in the input queue")
        self.parser.add_argument('--num_input_reader_threads', type=int, default=1,
            help="The number of threads for prefetching input data")
        self.parser.add_argument('--image_feature_name', type=str, default='image/data',
            help="The name of the SequenceExample context feature containing the image data")
        self.parser.add_argument('--caption_feature_name', type=str, default='image/caption_ids',
            help='The name of the SequenceExample feature list containing integer captions')
