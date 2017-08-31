import numpy as np
import tensorflow as tf


class TfTwoLayerNet(object):
    def __init__(self, input_size, output_size):
        self.learning_rate = 1e-4
        self.n_input = input_size
        self.n_hidden_1 = int(self.n_input * 2)
        self.n_hidden_2 = int(self.n_input)
        self.n_output = int(output_size)
        # TF model variables:
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Q = tf.placeholder("float", [None, self.n_output])
        self.beta = 0
        with tf.name_scope('layer_1'):
            W1 = tf.Variable(
                tf.random_normal([self.n_input, self.n_hidden_1],
                                 stddev=np.sqrt(2.0 / self.n_input)), name='W1')
            b1 = tf.Variable(tf.constant(0.1, shape=[self.n_hidden_1]), name='b1')
            h1 = tf.nn.relu(tf.add(tf.matmul(self.X, W1), b1))
        with tf.name_scope('layer_2'):
            W2 = tf.Variable(
                tf.random_normal([self.n_hidden_1, self.n_hidden_2],
                                 stddev=np.sqrt(2.0 / self.n_hidden_1)), name='W2')
            b2 = tf.Variable(tf.constant(0.1, shape=[self.n_hidden_2]), name='b2'),
            h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))
        with tf.name_scope('output_layer'):
            W3 = tf.Variable(
                tf.random_normal([self.n_hidden_2, self.n_output],
                                 stddev=np.sqrt(2.0 / self.n_hidden_2)), name='W3')
            b3 = tf.Variable(tf.constant(0.1, shape=[self.n_output]), name='b3')
            self.Q = tf.add(tf.matmul(h2, W3), b3)

        self.weights = [W1, b1, W2, b2, W3, b3]

        # Loss
        self.targetQ = tf.placeholder(tf.float32, [None])
        self.targetActionMask = tf.placeholder(
            tf.float32, [None, self.n_output])

        q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask),
                                 reduction_indices=[1])
        self.loss = (tf.reduce_mean(tf.square(tf.subtract(q_values, self.targetQ))) +
                     self.beta * tf.reduce_sum(tf.square(W1)) +
                     self.beta * tf.reduce_sum(tf.square(W2)) +
                     self.beta * tf.reduce_sum(tf.square(W3)))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def predict(self, x, weights=None):
        # Calculate prediction and ecoding
        if weights is None:
            q_values = self.sess.run(self.Q, feed_dict={self.X: x})
        else:
            feed_dict = {self.X: x}
            feed_dict.update(zip(self.weights, weights))
            q_values = self.sess.run(self.Q, feed_dict=feed_dict)
        return q_values

    def train(self, x, targetQ, targetActionMask):
        # Calculate next prediction, the modules encoding and the error of the last prediction
        # TODO: Optimize so the train step doesn't need to do a forward pass
        # Calculate next prediction, the modules encoding and the error of the last prediction
        feed_dict = {self.X: x,
                     self.targetQ: targetQ,
                     self.targetActionMask: targetActionMask}
        q_values, opt = self.sess.run([self.Q,
                                       self.optimizer],
                                      feed_dict=feed_dict)
        return

    def get_weights(self):
        return self.sess.run(self.weights)
