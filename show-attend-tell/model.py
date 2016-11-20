import tensorflow as tf

class Model(object):
    def __init__(self, args):
        # Helper functions
        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)

        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)

        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        self.x = tf.placeholder(tf.float32, shape=[None, args.input_length])
        self.y = tf.placeholder(tf.int64, shape=[None, args.seq_length])

        with tf.variable_scope('input_cnn'):
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])

            x_image = tf.reshape(self.x, [-1,28,28,1])

            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

        with tf.variable_scope('hidden_cnn'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        with tf.variable_scope('hidden_cnn2'):
            W_conv3 = weight_variable([5,5,64,64])
            b_conv3 = bias_variable([64])

            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)

        with tf.variable_scope('seq2seq-atten'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size)
            cell = tf.nn.run_cell.MultiRNNCell([cell] * args.num_layers)
            outputs, state = tf.nn.seq2seq.embedding_attention_decoder(all_inputs, initial_state, cell, num_encoder_symbols=1024, num_decoder_symbols=args.vocab_size, args.embedding_size)
