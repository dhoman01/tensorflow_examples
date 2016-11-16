import tensorflow as tf

class Model(object):
    def __init__(self, args):
        # Placeholders
        self.x = tf.placeholder(tf.float32, [None, args.num_steps, args.seq_length]);
        self.y = tf.placeholder(tf.float32, [None, args.num_classes]);

        x = self.x;
        y = self.y;

        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([args.num_layers, args.num_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([args.num_classes]))
        }

        # Reshape X
        x = tf.transpose(x, [1,0,2])
        x = tf.reshape(x, [-1, args.seq_length])
        x = tf.split(0, args.num_steps, x)

        with tf.variable_scope('lstm'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(args.num_layers, forget_bias=1.0)
            outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
            self.pred = tf.matmul(outputs[-1], self.weights['out'] + self.biases['out'])

        with tf.variable_scope('train'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(self.cost)

        with tf.variable_scope('eval'):
            self.correct_pred = tf.equal(tf.argmax(self.pred, 1),tf.argmax(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
