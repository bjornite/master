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
        q = layers.fully_connected(out, num_outputs=n_output, activation_fn=None, scope="output")
        return q

class CBTfTwoLayerNet(object):
    def __init__(self, input_size, output_size, learning_rate=1e-4, reg_beta=1e-6, log_dir="test_logs", number = None, sess = None):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.beta = reg_beta
        self.n_hiddens = [64]
        self.use_batch_norm = False
        self.use_dropout = False
        self.clip_gradients = True
        self.use_huber_loss = False
        self.normalizedSDG = False
        # TF model variables:
        self.n_input = input_size
        self.n_output = int(output_size)
        self.number = number
        self.sess = sess
        if sess is None:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
            #config.gpu_options.per_process_gpu_memory_fraction = 0.4
            self.sess = tf.Session(config=config)
        self.keep_prob = tf.placeholder_with_default(0.8, shape=())
        self.norm_factor = tf.placeholder("float", shape=())
        with tf.variable_scope("inputs"):
            self.X = tf.placeholder("float", [None, self.n_input], name="state")
            self.X_next = tf.placeholder("float", [None, self.n_input], name="obs")
            self.knowledge_reward = tf.placeholder("float", [None], name="knowledge_reward")
            self.targetActionMask = tf.placeholder(
                tf.float32, [None, self.n_output], name="targetAction")
        with tf.name_scope("module_{}".format(number)):
            self.Q = mlp(self.n_hiddens,
                         self.X,
                         self.n_output,
                         "policy_network_{}".format(number),
                         self.use_batch_norm,
                         self.use_dropout,
                         self.keep_prob)
            inputAndAction = tf.concat([self.X, self.targetActionMask], axis=1)
            self.prediction = mlp([64],
                                  inputAndAction,
                                  self.n_input,
                                  "prediction_module_{}".format(number),
                                  self.use_batch_norm,
                                  self.use_dropout,
                                  self.keep_prob)
            self.error_prediction = mlp([64],
                                        inputAndAction,
                                        1,
                                        "error_prediction_module_{}".format(number),
                                        self.use_batch_norm,
                                        self.use_dropout,
                                        self.keep_prob)

        self.weightnames = [x.name for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if x.name.endswith('weights:0')]
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.name_scope('prediction_loss_{}'.format(number)):
            # Euclidean distance for each prediction
            self.pred_error = tf.sqrt(
                tf.reduce_sum(
                    tf.square(
                        tf.subtract(self.prediction, self.X_next)),
                    reduction_indices=[1]))
            _, self.pred_err_var = tf.nn.moments(self.pred_error, axes=[0])
            # Prediction loss
            self.pred_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(
                        tf.subtract(self.prediction, self.X_next)),
                reduction_indices=[1]))
            self.pred_loss_regularized = (self.pred_loss) #TODO: regularization
        with tf.name_scope('error_prediction_loss_{}'.format(number)):
            self.error_prediction_error = tf.reduce_sum(tf.subtract(self.error_prediction,
                                                                    self.knowledge_reward),
                                                        reduction_indices=[1])
            self.error_prediction_loss = tf.reduce_mean(tf.square(self.error_prediction_error))
            self.error_prediction_loss_regularized = self.error_prediction_loss#TODO: regularize
        with tf.name_scope('policy_loss_{}'.format(number)):
            self.targetQ = tf.placeholder(tf.float32, [None], name="TargetQValues")
            self.q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask),
                                          reduction_indices=[1])

            self.td_errors = tf.subtract(self.q_values, self.targetQ)
            if self.use_huber_loss:
                self.qvalue_error = tf.reduce_mean(tf.where(tf.abs(self.td_errors) < 1.0,
                                                            tf.square(self.td_errors)*0.5,
                                                            tf.abs(self.td_errors) - 0.5))
            else:
                self.qvalue_error = tf.reduce_mean(tf.square(self.td_errors))
            self.policy_loss = (self.qvalue_error)  # TODO: Regularize

        tf.summary.scalar('mean_meta_error_{}'.format(number), self.error_prediction_loss)
        tf.summary.scalar('mean qvalue_error_{}'.format(number), self.qvalue_error)
        tf.summary.scalar('mean_policy_loss_{}'.format(number), self.policy_loss)
        tf.summary.scalar('mean_pred_error_{}'.format(number), self.pred_loss)
        tf.summary.scalar('mean_pred_error_variance_{}'.format(number), self.pred_err_var)
        tf.summary.scalar('mean_q_value_{}'.format(number), tf.reduce_mean(self.q_values))

        with tf.name_scope('loss'):
            self.loss = (self.policy_loss +
                         self.pred_loss_regularized +
                         self.error_prediction_loss_regularized)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Compute the gradients for a list of variables.
        gradients = optimizer.compute_gradients(self.loss, self.var_list)
        grads = []
        # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        with tf.name_scope("gradient_clipping"):
            if self.normalizedSDG:
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        if "policy_network/output" not in var.name:
                            grads[i] = tf.scalar_mul(self.norm_factor, grad)
                        else:
                            print(var)
            if self.clip_gradients:
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        grads.append((tf.clip_by_norm(grad, 10), var))
            #grads, _ = tf.clip_by_global_norm(grads, 10.0)
        # Ask the optimizer to apply the capped gradients.
        #grads_and_vars = list(zip(grads, self.var_list))
        if grads == []:
            grads = gradients
        self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
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

    def train(self, x, k_rew, x_next, targetQ, targetActionMask, normalization_factor, no_tf_log):
        # Calculate next prediction, the modules encoding and the error of the last prediction
        # TODO: Optimize so the train step doesn't need to do a forward pass
        feed_dict = {self.X: x,
                     self.knowledge_reward: np.array(k_rew),
                     self.X_next: x_next,
                     self.targetQ: targetQ,
                     self.targetActionMask: targetActionMask,
                     self.norm_factor: normalization_factor}
        opt, summary, global_step, td_errors = self.sess.run([self.train_op,
                                                              self.merged,
                                                              self.global_step,
                                                              self.td_errors],
                                                             feed_dict=feed_dict)
        if global_step % 100 == 0 and not no_tf_log:
            self.train_writer.add_summary(summary, global_step)
        return td_errors

    def get_prediction_error(self, x, a, x_next):
        feed_dict = {self.X: x,
                     self.targetActionMask: a,
                     self.X_next: x_next}
        return self.sess.run(self.pred_error,
                             feed_dict=feed_dict)

    def get_meta_prediction_error(self, x, a, knowledge_rewards, x_next):
        feed_dict = {self.X: x,
                     self.targetActionMask: a,
                     self.knowledge_reward: knowledge_rewards,
                     self.X_next: x_next}
        return self.sess.run(self.error_prediction_error,
                             feed_dict=feed_dict)

    def get_weights(self):
        return self.sess.run(self.weights)


class ModularNet(object):

    def __init__(self, input_size, output_size, learning_rate=1e-4, reg_beta=1e-6, log_dir="test_logs", sess = None):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.beta = reg_beta
        self.n_hiddens = [64]
        self.use_batch_norm = False
        self.use_dropout = False
        self.clip_gradients = True
        self.use_huber_loss = False
        self.normalizedSDG = False
        # TF model variables:
        self.n_input = input_size
        self.n_output = int(output_size)
        self.sess = sess
        if sess is None:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
            #config.gpu_options.per_process_gpu_memory_fraction = 0.4
            self.sess = tf.Session(config=config)
        self.keep_prob = tf.placeholder_with_default(0.8, shape=())
        self.norm_factor = tf.placeholder("float", shape=())
        self.log_dir = log_dir
        with tf.variable_scope("inputs"):
            self.X = tf.placeholder("float", [None, self.n_input], name="state")
            #self.mask = tf.placeholder("float", [self.n_input], name="mask")
            self.X_next = tf.placeholder("float", [None, self.n_input], name="obs")
            self.knowledge_reward = tf.placeholder("float", [None], name="knowledge_reward")
            self.targetActionMask = tf.placeholder(
                tf.float32, [None, self.n_output], name="targetAction")
        with tf.variable_scope("targets"):
            self.targetQ = tf.placeholder(tf.float32, [None], name="TargetQValues")
        self.train_writer = tf.summary.FileWriter(self.log_dir,
                                                  self.sess.graph)

    def make_graph_module(self, number):
        # Build required graph structure
        with tf.name_scope("module_{}".format(number)):
            #maskedInput = tf.multiply(self.X, self.mask)
            Q = mlp(self.n_hiddens,
                    self.X,
                    self.n_output,
                    "module_{}_q_network".format(number),
                    self.use_batch_norm,
                    self.use_dropout,
                    self.keep_prob)
            inputAndAction = tf.concat([self.X, self.targetActionMask], axis=1)
            prediction = mlp([64],
                             inputAndAction,
                             self.n_input,
                             "module_{}_prediction_network".format(number),
                             self.use_batch_norm,
                             self.use_dropout,
                             self.keep_prob)
            error_prediction = mlp([64],
                                   inputAndAction,
                                   1,
                                   "module_{}_error_prediction".format(number),
                                   self.use_batch_norm,
                                   self.use_dropout,
                                   self.keep_prob)

            weightnames = [x.name for x in
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope="module_{}".format(number))
                           if x.name.endswith('weights:0')]
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope="module_{}".format(number))
            with tf.variable_scope('prediction_loss_{}'.format(number)):
                # Prediction loss
                pred_error = tf.reduce_sum(tf.square(tf.subtract(prediction, self.X_next)),
                                           reduction_indices=[1])
                pred_loss = tf.reduce_mean(pred_error)
                pred_loss_regularized = (pred_loss) #TODO: regularization
            with tf.name_scope('error_prediction_loss_{}'.format(number)):
                error_prediction_error = tf.reduce_sum(tf.subtract(error_prediction,
                                                                   self.knowledge_reward),
                                                       reduction_indices=[1])
                error_prediction_loss = tf.reduce_mean(tf.square(error_prediction_error))
                error_prediction_loss_regularized = error_prediction_loss#TODO: regularize

            with tf.variable_scope('policy_loss_{}'.format(number)):
                q_values = tf.reduce_sum(tf.multiply(Q, self.targetActionMask),
                                         reduction_indices=[1])
                td_errors = tf.subtract(q_values, self.targetQ)
                if self.use_huber_loss:
                    qvalue_error = tf.reduce_mean(tf.where(tf.abs(td_errors) < 1.0,
                                                           tf.square(td_errors)*0.5,
                                                           tf.abs(td_errors) - 0.5))
                else:
                    qvalue_error = tf.reduce_mean(tf.square(td_errors))
                policy_loss = (qvalue_error)  # TODO: Regularize

            tf.summary.scalar('mean qvalue_error_{}'.format(number), qvalue_error)
            tf.summary.scalar('mean_policy_loss_{}'.format(number), policy_loss)
            tf.summary.scalar('mean_pred_error_{}'.format(number), pred_loss)
            tf.summary.scalar('mean_q_value_{}'.format(number), tf.reduce_mean(q_values))

            with tf.variable_scope('loss'):
                loss = (policy_loss + pred_loss_regularized + error_prediction_loss_regularized)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            local_step = tf.Variable(0, name='local_step', trainable=False)

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope="module_{}".format(number))
            # Compute the gradients for a list of variables.
            gradients = optimizer.compute_gradients(loss, var_list)
            grads = []
            # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
            # need to the 'gradient' part, for example cap them, etc.
            with tf.name_scope("gradient_clipping"):
                if self.normalizedSDG:
                    for i, (grad, var) in enumerate(gradients):
                        if grad is not None:
                            if "policy_network/output" not in var.name:
                                grads[i] = tf.scalar_mul(self.norm_factor, grad)
                            else:
                                print(var)
                if self.clip_gradients:
                    for i, (grad, var) in enumerate(gradients):
                        if grad is not None:
                            grads.append((tf.clip_by_norm(grad, 10), var))

            # Ask the optimizer to apply the capped gradients.

            if grads == []:
                grads = gradients
            train_op = optimizer.apply_gradients(grads, global_step=local_step)
            current_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope="module_{}".format(number))

            merged = tf.summary.merge_all()
            #temp = set(tf.all_variables())
            #self.sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
            #new_vars = list(tf.get_variable(name) for name in
            #                tf.global_variables(scope="module_{}".format(number)))
            #print(new_vars)
            #self.sess.run(tf.variables_initializer(new_vars))
            # Return functions for the necessary operations
            return Q, train_op, merged, local_step, td_errors, pred_error, error_prediction_error, weights, self.train_writer

    def init(self):
        init = tf.variables_initializer(tf.global_variables())
        self.sess.run(init)
        self.train_writer = tf.summary.FileWriter(self.log_dir,
                                                  self.sess.graph)

    def make_module(self, number):
        Q, train_op, merged, local_step, td_errors, pred_error, error_prediction_error, weights, train_writer = self.make_graph_module(number)

        def predict(self, x, input_weights=None):
            # Calculate prediction and ecoding
            if input_weights is None:
                q_values = self.sess.run(Q, feed_dict={self.X: x,
                                                       self.keep_prob: 1.0})
            else:
                feed_dict = {self.X: x, self.keep_prob: 1.0}
                feed_dict.update(zip(weights, input_weights))
                q_values = self.sess.run(Q, feed_dict=feed_dict)
            return q_values

        def train(self, x, k_rew, x_next, targetQ, targetActionMask, normalization_factor, no_tf_log):
            # Calculate next prediction, the modules encoding and the error of the last prediction
            # TODO: Optimize so the train step doesn't need to do a forward pass
            feed_dict = {self.X: x,
                         self.knowledge_reward: np.array(k_rew),
                         self.X_next: x_next,
                         self.targetQ: targetQ,
                         self.targetActionMask: targetActionMask,
                         self.norm_factor: normalization_factor}
            opt, summary, step, td_err = self.sess.run([train_op,
                                                        merged,
                                                        local_step,
                                                        td_errors],
                                                       feed_dict=feed_dict)
            if step % 100 == 0 and not no_tf_log:
                train_writer.add_summary(summary, step)
            return td_err

        def get_prediction_error(self, x, a, x_next):
            feed_dict = {self.X: x,
                         self.targetActionMask: a,
                         self.X_next: x_next}
            return self.sess.run(pred_error,
                                 feed_dict=feed_dict)

        def get_meta_prediction_error(self, x, a, knowledge_rewards, x_next):
            feed_dict = {self.X: x,
                         self.targetActionMask: a,
                         self.knowledge_reward: np.array(knowledge_rewards),
                         self.X_next: x_next}
            return self.sess.run(error_prediction_error,
                                 feed_dict=feed_dict)
        
        def get_weights(self):
            return self.sess.run(weights)

        def assign_weights(self, input_weights):
            feed_dict = {}
            feed_dict.update(zip(weights, input_weights))
            for name, value in feed_dict.items():
                self.sess.run(tf.assign(name, value))

        return self, predict, train, get_prediction_error, get_meta_prediction_error, get_weights, assign_weights
