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

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

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

        with tf.variable_scope('densely_connected_nn'):
            W_fc1 = weight_variable([8 * 8 * 16, 1024])
            b_fc1 = bias_variable([1024])

            h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*16])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

            self.keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        with tf.variable_scope('read_out_layer'):
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])

            self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        with tf.variable_scope('train'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_conv, self.y))
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.variable_scope('eval'):
            self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
