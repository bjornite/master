import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def mlp(n_hiddens, inpt, n_output, scope, use_batch_norm=False, use_dropout=False, keep_prob=0.8):
    with tf.variable_scope(scope, reuse=False):
        out = inpt
        for h in n_hiddens:
            out = layers.fully_connected(out, num_outputs=h, activation_fn=None)
            if use_batch_norm:
                out = tf.layers.batch_normalization(out)
            out = tf.nn.relu(out)
            if use_dropout:
                out = tf.nn.dropout(out, keep_prob, noise_shape=[1, h])
        q = layers.fully_connected(out, num_outputs=n_output, activation_fn=None)
        return q

class CBTfTwoLayerNet(object):
    def __init__(self, input_size, output_size, learning_rate=1e-4, reg_beta=1e-6, log_dir="test_logs"):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.beta = reg_beta
        self.keep_prob = tf.placeholder_with_default(0.8, shape=())
        self.n_hiddens = [64]
        self.use_batch_norm = False
        self.use_dropout = False
        self.clip_gradients = True
        self.use_huber_loss = False
        # TF model variables:
        self.n_input = input_size
        self.n_output = int(output_size)
        self.X = tf.placeholder("float", [None, self.n_input], name="state")
        self.X_next = tf.placeholder("float", [None, self.n_input], name="obs")
        self.knowledge_reward = tf.placeholder("float", [None], name="knowledge_reward")
        self.targetActionMask = tf.placeholder(
            tf.float32, [None, self.n_output])

        self.Q = mlp(self.n_hiddens,
                     self.X,
                     self.n_output,
                     "policy_network",
                     self.use_batch_norm,
                     self.use_dropout,
                     self.keep_prob)
        inputAndAction = tf.concat([self.X, self.targetActionMask], axis=1)
        self.prediction = mlp([64],
                              inputAndAction,
                              self.n_input,
                              "prediction_module",
                              self.use_batch_norm,
                              self.use_dropout,
                              self.keep_prob)
        self.error_prediction = mlp([64],
                                    inputAndAction,
                                    1,
                                    "error_prediction_module",
                                    self.use_batch_norm,
                                    self.use_dropout,
                                    self.keep_prob)

        self.weightnames = [x.name for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if x.name.endswith('weights:0')]
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        with tf.name_scope('prediction_loss'):
            # Prediction loss
            self.pred_error = tf.reduce_sum(tf.square(tf.subtract(self.prediction, self.X_next)),
                                            reduction_indices=[1])

            self.pred_loss = tf.reduce_mean(self.pred_error)
            self.pred_loss_regularized = (self.pred_loss) #TODO: regularization
        with tf.name_scope('error_prediction_loss'):
            self.error_prediction_error = tf.reduce_sum(tf.subtract(self.error_prediction,
                                                                    self.knowledge_reward),
                                                        reduction_indices=[1])
            self.error_prediction_loss = tf.reduce_mean(self.error_prediction_error)
            self.error_prediction_loss_regularized = (tf.abs(self.error_prediction_loss)) #TODO: regularize
        with tf.name_scope('policy_loss'):
            self.targetQ = tf.placeholder(tf.float32, [None], name="TargetQValues")

            self.q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask),
                                          reduction_indices=[1])

            diff = tf.subtract(self.q_values, self.targetQ)
            if self.use_huber_loss:
                self.qvalue_error = tf.reduce_mean(tf.where(tf.abs(diff) < 1.0,
                                                            tf.square(diff)*0.5,
                                                            tf.abs(diff) - 0.5))
            else:
                self.qvalue_error = tf.reduce_mean(tf.square(diff))
            self.policy_loss = (self.qvalue_error)  # TODO: Regularize

        tf.summary.scalar('mean_meta_error', self.error_prediction_loss)
        tf.summary.scalar('mean qvalue_error', self.qvalue_error)
        tf.summary.scalar('mean_policy_loss', self.policy_loss)
        tf.summary.scalar('mean_pred_error', self.pred_loss)
        tf.summary.scalar('mean_q_value', tf.reduce_mean(self.q_values))

        with tf.name_scope('loss'):
            self.loss = (self.policy_loss +
                         self.pred_loss_regularized +
                         self.error_prediction_loss_regularized)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Compute the gradients for a list of variables.
        grads = optimizer.compute_gradients(self.loss, self.var_list)

        # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        if self.clip_gradients:
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    grads[i] = (tf.clip_by_norm(grad, 10), var)
            #grads, _ = tf.clip_by_global_norm(grads, 10.0)

        # Ask the optimizer to apply the capped gradients.
        #grads_and_vars = list(zip(grads, self.var_list))
        self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        config = tf.ConfigProto(
             device_count={'GPU': 1}
        )
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)
        self.train_writer = tf.summary.FileWriter(log_dir,
                                                  self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(init)

    def predict(self, x, weights=None):
        # Calculate prediction and ecoding
        if weights is None:
            q_values = self.sess.run(self.Q, feed_dict={self.X: x,
                                                        self.keep_prob: 1.0})
        else:
            feed_dict = {self.X: x, self.keep_prob: 1.0}
            feed_dict.update(zip(self.weights, weights))
            q_values = self.sess.run(self.Q, feed_dict=feed_dict)
        return q_values

    def get_q_value_uncertainty(self, x):
        # Calculate prediction and ecoding
        q_values = []
        for i in range(10):
            q_values.append(self.sess.run(self.Q, feed_dict={self.X: x,
                                                             self.keep_prob: 0.8}))
        return np.std(q_values, axis=0)

    def get_prediction_uncertainty(self, x, a):
        # Calculate prediction and ecoding
        num_preds = 10
        predictions = []
        act = np.zeros(self.n_output)
        act[a] = 1
        for i in range(num_preds):
            predictions.append(self.sess.run(self.prediction, feed_dict={self.X: [x],
                                                                         self.targetActionMask: [act],
                                                                         self.keep_prob: 0.8}))
        return np.mean(np.std(predictions, axis=0))

    def train(self, x, k_rew, x_next, targetQ, targetActionMask, no_tf_log):
        # Calculate next prediction, the modules encoding and the error of the last prediction
        # TODO: Optimize so the train step doesn't need to do a forward pass
        feed_dict = {self.X: x,
                     self.knowledge_reward: np.array(k_rew),
                     self.X_next: x_next,
                     self.targetQ: targetQ,
                     self.targetActionMask: targetActionMask}
        opt, summary, global_step = self.sess.run([self.train_op,
                                                   self.merged,
                                                   self.global_step],
                                                  feed_dict=feed_dict)
        if global_step % 100 == 0 and not no_tf_log:
            self.train_writer.add_summary(summary, global_step)
        return

    def get_prediction_error(self, x, a, x_next):
        feed_dict = {self.X: x,
                     self.targetActionMask: a,
                     self.X_next: x_next}
        return self.sess.run(self.pred_error,
                             feed_dict=feed_dict)

    def get_meta_prediction_error(self, x, a, knowledge_rewards, x_next):
        feed_dict = {self.X: x,
                     self.targetActionMask: a,
                     self.knowledge_reward: np.array(knowledge_rewards),
                     self.X_next: x_next}
        return self.sess.run(self.error_prediction_error,
                             feed_dict=feed_dict)

    def get_weights(self):
        return self.sess.run(self.weights)
