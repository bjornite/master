import numpy as np
import tensorflow as tf

class CBTfTwoLayerNet(object):
    def __init__(self, input_size, output_size, log_dir="test_logs"):
        # Hyperparameters
        self.learning_rate = 1e-3
        self.beta = 1e-6
        self.n_hidden_1 = 128
        self.n_hidden_2 = 128

        # TF model variables:
        self.n_input = input_size
        self.n_output = int(output_size)
        self.X = tf.placeholder("float", [None, self.n_input], name="state")
        self.X_next = tf.placeholder("float", [None, self.n_input], name="obs")
        self.action = tf.placeholder("float", [None, self.n_output], name="action")
        self.knowledge_reward = tf.placeholder("float", [None], name="knowledge_reward")

        with tf.name_scope('policy_network'):
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
        with tf.name_scope('prediction_layer'):
            WP = tf.Variable(
                tf.random_normal([self.n_input, self.n_input],
                                 stddev=np.sqrt(2.0 / self.n_input)), name='WP')
            bP = tf.Variable(tf.constant(0.1, shape=[self.n_input]), name='bP')
            self.prediction = tf.add(tf.matmul(self.X, WP), bP)
        with tf.name_scope('error_prediction_layer'):
            WEP = tf.Variable(
                tf.random_normal([self.n_input, 1],
                                 stddev=np.sqrt(2.0 / self.n_input)), name='WEP')
            bEP = tf.Variable(tf.constant(0.1, shape=[1]), name='bEP')
            self.error_prediction = tf.add(tf.matmul(self.X, WEP), bEP)

        self.weightnames = ["W1", "b1",
                            "W2", "b2",
                            "W3", "b3",
                            "WP", "bP",
                            "WEP", "bEP"]
        self.weights = [W1, b1,
                        W2, b2,
                        W3, b3,
                        WP, bP,
                        WEP, bEP]

        with tf.name_scope('prediction_loss'):
            # Prediction loss
            self.pred_error = tf.reduce_sum(tf.square(tf.subtract(self.prediction, self.X_next)),
                                            reduction_indices=[1])
            # self.pred_error = tf.divide(unnormalized_pred_error,
            #                            tf.reduce_max(unnormalized_pred_error, axis=[0]))
            self.pred_loss = tf.reduce_mean(self.pred_error)
            self.pred_loss_regularized = (self.pred_loss +
                                          self.beta * tf.reduce_sum(tf.square(WP)))
        with tf.name_scope('error_prediction_loss'):
            # TODO: Figure out how to normalize this, get huge (and negative) values
            self.error_prediction_error = tf.reduce_sum(tf.subtract(self.error_prediction,
                                                                    self.knowledge_reward),
                                                        reduction_indices=[1])
            self.error_prediction_loss = tf.reduce_mean(self.error_prediction_error)
            self.error_prediction_loss_regularized = (tf.abs(self.error_prediction_loss) +
                                                      self.beta * tf.reduce_sum(tf.square(WEP)))
        with tf.name_scope('policy_loss'):
            # Loss
            self.targetQ = tf.placeholder(tf.float32, [None], name="TargetQValues")
            self.targetActionMask = tf.placeholder(
                tf.float32, [None, self.n_output])
            q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask),
                                     reduction_indices=[1])
            self.qvalue_error = tf.reduce_mean(tf.square(tf.subtract(q_values, self.targetQ)))
            self.policy_loss = (self.qvalue_error +
                                self.beta * tf.reduce_sum(tf.square(W1)) +
                                self.beta * tf.reduce_sum(tf.square(W2)) +
                                self.beta * tf.reduce_sum(tf.square(W3)))
        for w in range(len(self.weights)):
            tf.summary.histogram(self.weightnames[w], self.weights[w])
        tf.summary.histogram('meta_error_prediction', self.error_prediction_error)
        tf.summary.histogram('pred_error', self.pred_error)
        tf.summary.scalar('mean_meta_error', self.error_prediction_loss)
        tf.summary.scalar('mean qvalue_error', self.qvalue_error)
        tf.summary.scalar('mean_policy_loss', self.policy_loss)
        tf.summary.scalar('mean_pred_error', self.pred_loss)

        with tf.name_scope('loss'):
            self.loss = (self.policy_loss +
                         self.pred_loss_regularized +
                         self.error_prediction_loss_regularized)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Compute the gradients for a list of variables.
        grads = tf.gradients(self.loss, self.var_list)

        # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        grads, _ = tf.clip_by_global_norm(grads, 40.0)

        # Ask the optimizer to apply the capped gradients.
        grads_and_vars = list(zip(grads, self.var_list))
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        self.merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.Session(config=config)
        self.train_writer = tf.summary.FileWriter(log_dir,
                                                  self.sess.graph)
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

    def train(self, x, k_rew, x_next, targetQ, targetActionMask):
        # Calculate next prediction, the modules encoding and the error of the last prediction
        # TODO: Optimize so the train step doesn't need to do a forward pass
        # Calculate next prediction, the modules encoding and the error of the last prediction
        feed_dict = {self.X: x,
                     self.knowledge_reward: np.array(k_rew),
                     self.X_next: x_next,
                     self.targetQ: targetQ,
                     self.targetActionMask: targetActionMask}
        opt, summary, global_step = self.sess.run([self.train_op,
                                                   self.merged,
                                                   self.global_step],
                                                  feed_dict=feed_dict)
        if global_step % 100 == 0:
            self.train_writer.add_summary(summary, global_step)
        return

    def get_prediction_error(self, x, x_next):
        feed_dict = {self.X: x,
                     self.X_next: x_next}
        return self.sess.run(self.pred_error,
                             feed_dict=feed_dict)

    def get_meta_prediction_error(self, x, knowledge_rewards, x_next):
        feed_dict = {self.X: x,
                     self.knowledge_reward: np.array(knowledge_rewards),
                     self.X_next: x_next}
        return self.sess.run(self.error_prediction_error,
                             feed_dict=feed_dict)

    def get_weights(self):
        return self.sess.run(self.weights)
