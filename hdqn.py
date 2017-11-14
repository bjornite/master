from Agent import Agent
import numpy as np
import tensorflow as tf
import random
import copy

class HDQN(Agent):
    def __init__(self, name, env, log_dir):
        super(HDQN, self).__init__(name, env, log_dir)
        self.num_layers = 3
        self.state_size = self.observation_space.shape[0]
        self.z_sizes = [self.state_size*2, 4, 3, 2]
        self.num_policies = [self.action_space.n, 5, 10, 1]
        self.accumulated_rewards = [0]*self.num_layers
        config = tf.ConfigProto(
             device_count={'GPU': 1}
        )
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)
        self.layers = []
        for m in range(self.num_layers):
            self.layers.append(Layer(m,
                                     self.num_policies[m+1],
                                     self.z_sizes[m],
                                     self.num_policies[m],
                                     self.z_sizes[m+1],
                                     self.sess))
        self.timestep = 0
        self.z_values = [[]]*self.num_layers
        self.merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        self.train_writer = tf.summary.FileWriter(log_dir,
                                                  self.sess.graph)
        self.sess.run(init)


    def train(self, sars_tuple):
        s, a, s_next, r, done = sars_tuple
        # Train each relevant layer, save values for the higher layers
        # Train policies
        # Train z-networks
        z = self.z_values[0]
        self.layers[0].train_policy(z, a, r, done)
        self.layers[0].train_z(s, a, s_next)
        z_next = self.layers[0].get_z(s_next)
        for m in range(1, self.num_layers):
            if self.timestep % pow(2, m) == 0:
                z = self.z_values[m-1]
                a = self.layers[m].current_policy
                r = self.accumulated_rewards[m]
                self.layers[m].train_policy(z, a, r, done)
                z_next = self.layers[m].get_z(z_next)
                self.layers[m].train_z(z, a, z_next)
                self.accumulated_rewards[m] = 0
            else:
                self.accumulated_rewards[m] += r

    def get_action(self, observation, is_test=False):
        # If timestep % 2^m == 0, run interactions up to layer m
        # Evaluate bottom layer
        self.z_values[0] = self.layers[0].get_z(observation)
        # Get z-values up the hierarchy
        for m in range(1, self.num_layers):
            if self.timestep % pow(2, m) == 0:
                self.z_values[m] = self.layers[m].get_z(self.z_values[m-1])
        # Apply actions down the hierarchy
        for m in range(self.num_layers-1, 0, -1):
            if self.timestep % pow(2, m) == 0:
                self.layers[m-1].policy_weights = self.layers[m].get_actions(self.z_values[m])
        action = np.argmax(self.layers[0].get_actions(self.z_values[0])[0])
        self.timestep += 1
        return action

class Layer():
    def __init__(self, layer_number, num_policies, obs_shape, act_shape, z_shape, sess):
        self.sess = sess
        self.num_policies = num_policies
        self.current_policy = 0
        self.n_input = obs_shape
        self.n_output = act_shape
        self.n_z = z_shape
        self.policy_weights = [[1.0/num_policies] * num_policies]
        self.policies = []
        self.z_network = Policy(layer_number, -1, self.n_input, self.n_z, self.sess)
        for i in range(self.num_policies):
            p = Policy(layer_number, i, self.n_z, self.n_output, self.sess)
            self.policies.append(p)

    def get_z(self, observation):
        # return z for this layer
        return self.z_network.forward(observation)[0].reshape([-1])

    def get_actions(self, z):
        # return list of action selection probabilities for the most confident policy
        self.current_policy = np.argmax(self.policy_weights[0])
        action, confidence, n = self.policies[self.current_policy].forward(z)
        return action

    def train_z(self, s, next_s, a):
        # Do update step on z-network
        pass

    def train_policy(self, n, z, targetQ, targetActionMask):
        # Do update step on policy number n
        pass


class Policy():
    def __init__(self, layer, n,  n_input, n_output, sess):
        self.sess = sess
        self.keep_prob = tf.placeholder_with_default(0.8, shape=())
        self.learning_rate = 1e-4
        self.n_hidden_1 = 4
        self.n_hidden_2 = 4
        self.policy_n = n
        # TF model variables:
        self.n_input = n_input
        self.n_output = int(n_output)
        self.z = tf.placeholder("float", [None, self.n_input], name="state")
        self.targetActionMask = tf.placeholder(
            tf.float32, [None, self.n_output])

        self.namescope = 'policy_network_{0}_{1}'.format(layer, n)

        with tf.name_scope(self.namescope):
            with tf.name_scope('layer_1'):
                self.W1 = tf.Variable(
                    tf.random_normal([self.n_input, self.n_hidden_1],
                                     stddev=np.sqrt(2.0 / self.n_input)), name='W1')
                self.b1 = tf.Variable(tf.constant(0.1, shape=[self.n_hidden_1]), name='b1')
                self.z1 = tf.add(tf.matmul(self.z, self.W1), self.b1)
                self.bn_input_1 = tf.layers.batch_normalization(self.z1)
                self.a1 = tf.nn.relu(self.bn_input_1)
                self.h1 = tf.nn.dropout(self.a1, self.keep_prob, noise_shape=[1, self.n_hidden_1])
            with tf.name_scope('layer_2'):
                self.W2 = tf.Variable(
                    tf.random_normal([self.n_hidden_1, self.n_hidden_2],
                                     stddev=np.sqrt(2.0 / self.n_hidden_1)), name='W2')
                self.b2 = tf.Variable(tf.constant(0.1, shape=[self.n_hidden_2]), name='b2'),
                self.z2 = tf.add(tf.matmul(self.h1, self.W2), self.b2)
                self.bn_input_2 = tf.layers.batch_normalization(self.z2)
                self.a2 = tf.nn.relu(self.bn_input_2)
                self.h2 = tf.nn.dropout(self.a2, self.keep_prob, noise_shape=[1, self.n_hidden_2])
            with tf.name_scope('output_layer'):
                self.W3 = tf.Variable(
                    tf.random_normal([self.n_hidden_2, self.n_output],
                                     stddev=np.sqrt(2.0 / self.n_hidden_2)), name='W3')
                self.b3 = tf.Variable(tf.constant(0.1, shape=[self.n_output]), name='b3')
                self.Q = tf.add(tf.matmul(self.h2, self.W3), self.b3)
                self.probs = tf.nn.softmax(self.Q)
            with tf.name_scope('policy_loss'):
                # Loss
                self.targetQ = tf.placeholder(tf.float32, [None], name="TargetQValues")
                
                self.q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask),
                                              reduction_indices=[1])
                
                self.qvalue_error = tf.reduce_mean(tf.square(tf.subtract(self.q_values, self.targetQ)))
                self.policy_loss = self.qvalue_error

                tf.summary.scalar('mean_policy_loss', self.policy_loss)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_step = tf.Variable(0, name='train_step', trainable=False)

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # Compute the gradients for a list of variables.
            grads = tf.gradients(self.policy_loss, self.var_list)

            # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
            # need to the 'gradient' part, for example cap them, etc.
            grads, _ = tf.clip_by_global_norm(grads, 5.0)

            # Ask the optimizer to apply the capped gradients.
            grads_and_vars = list(zip(grads, self.var_list))
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.train_step)

    def get_q_value_uncertainty(self, x):
        # Calculate prediction and ecoding
        q_values = []
        for i in range(10):
            q_values.append(self.sess.run(self.Q, feed_dict={self.z: [x],
                                                             self.keep_prob: 0.8}))
        return np.std(q_values, axis=0)

    def forward(self, s, weights=None):
        # Calculate prediction and ecoding
        if weights is None:
            probs = self.sess.run(self.probs, feed_dict={self.z: [s],
                                                         self.keep_prob: 1.0})
        else:
            feed_dict = {self.z: [s], self.keep_prob: 1.0}
            feed_dict.update(zip(self.weights, weights))
            probs = self.sess.run(self.probs, feed_dict=feed_dict)
        confidence = np.mean(self.get_q_value_uncertainty(s))
        return probs, confidence, self.policy_n


    def train(z, targetQ, targetActionMask, no_tf_log):
        # Calculate next prediction, the modules encoding and the error of the last prediction
        # TODO: Optimize so the train step doesn't need to do a forward pass
        feed_dict = {self.z: z,
                     self.targetQ: targetQ,
                     self.targetActionMask: targetActionMask}
        opt, summary, global_step = self.sess.run([self.train_op,
                                                   self.merged,
                                                   self.global_step],
                                                  feed_dict=feed_dict)
        if global_step % 100 == 0 and not no_tf_log:
            self.train_writer.add_summary(summary, global_step)
        return

class Z_network():
    def __init__(self, layer, n,  n_input, n_output, sess):
        self.sess = sess
        self.keep_prob = tf.placeholder_with_default(0.8, shape=())
        self.learning_rate = 1e-4
        self.n_hidden_1 = 4
        self.n_hidden_2 = 4
        self.policy_n = n
        # TF model variables:
        self.n_input = n_input
        self.n_output = int(n_output)
        self.s = tf.placeholder("float", [None, self.n_input], name="state")
        self.next_z = tf.placeholder("float", [None, self.n_input], name="obs")
        self.targetActionMask = tf.placeholder(
            tf.float32, [None, self.n_output])

        self.namescope = 'z_network_{0}'.format(layer)

        with tf.name_scope(self.namescope):
            with tf.name_scope('layer_1'):
                self.W1 = tf.Variable(
                    tf.random_normal([self.n_input, self.n_hidden_1],
                                     stddev=np.sqrt(2.0 / self.n_input)), name='W1')
                self.b1 = tf.Variable(tf.constant(0.1, shape=[self.n_hidden_1]), name='b1')
                self.z1 = tf.add(tf.matmul(self.z, self.W1), self.b1)
                self.bn_input_1 = tf.layers.batch_normalization(self.z1)
                self.a1 = tf.nn.relu(self.bn_input_1)
                self.h1 = tf.nn.dropout(self.a1, self.keep_prob, noise_shape=[1, self.n_hidden_1])
            with tf.name_scope('layer_2'):
                self.W2 = tf.Variable(
                    tf.random_normal([self.n_hidden_1, self.n_hidden_2],
                                     stddev=np.sqrt(2.0 / self.n_hidden_1)), name='W2')
                self.b2 = tf.Variable(tf.constant(0.1, shape=[self.n_hidden_2]), name='b2'),
                self.z2 = tf.add(tf.matmul(self.h1, self.W2), self.b2)
                self.bn_input_2 = tf.layers.batch_normalization(self.z2)
                self.a2 = tf.nn.relu(self.bn_input_2)
                self.h2 = tf.nn.dropout(self.a2, self.keep_prob, noise_shape=[1, self.n_hidden_2])
            with tf.name_scope('output_layer'):
                self.W3 = tf.Variable(
                    tf.random_normal([self.n_hidden_2, self.n_output],
                                     stddev=np.sqrt(2.0 / self.n_hidden_2)), name='W3')
                self.b3 = tf.Variable(tf.constant(0.1, shape=[self.n_output]), name='b3')
                self.z = tf.add(tf.matmul(self.h2, self.W3), self.b3)
                self.probs = tf.nn.softmax(self.Q)
            with tf.name_scope('z_loss'):
                # Loss
                self.targetQ = tf.placeholder(tf.float32, [None], name="TargetQValues")
                
                self.z_loss = tf.reduce_sum(tf.subtract(self.probs, self.targetActionMask),
                                              reduction_indices=[1])
                
                self.qvalue_error = tf.reduce_mean(tf.square(tf.subtract(self.q_values, self.targetQ)))
                self.policy_loss = self.qvalue_error

                tf.summary.scalar('mean_policy_loss', self.policy_loss)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_step = tf.Variable(0, name='train_step', trainable=False)

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # Compute the gradients for a list of variables.
            grads = tf.gradients(self.policy_loss, self.var_list)

            # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
            # need to the 'gradient' part, for example cap them, etc.
            grads, _ = tf.clip_by_global_norm(grads, 5.0)

            # Ask the optimizer to apply the capped gradients.
            grads_and_vars = list(zip(grads, self.var_list))
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.train_step)

    def get_q_value_uncertainty(self, x):
        # Calculate prediction and ecoding
        q_values = []
        for i in range(10):
            q_values.append(self.sess.run(self.Q, feed_dict={self.z: [x],
                                                             self.keep_prob: 0.8}))
        return np.std(q_values, axis=0)

    def forward(self, s, weights=None):
        # Calculate prediction and ecoding
        if weights is None:
            probs = self.sess.run(self.probs, feed_dict={self.z: [s],
                                                         self.keep_prob: 1.0})
        else:
            feed_dict = {self.z: [s], self.keep_prob: 1.0}
            feed_dict.update(zip(self.weights, weights))
            probs = self.sess.run(self.probs, feed_dict=feed_dict)
        confidence = np.mean(self.get_q_value_uncertainty(s))
        return probs, confidence, self.policy_n


    def train(z, targetQ, targetActionMask, no_tf_log):
        # Calculate next prediction, the modules encoding and the error of the last prediction
        # TODO: Optimize so the train step doesn't need to do a forward pass
        feed_dict = {self.z: z,
                     self.targetQ: targetQ,
                     self.targetActionMask: targetActionMask}
        opt, summary, global_step = self.sess.run([self.train_op,
                                                   self.merged,
                                                   self.global_step],
                                                  feed_dict=feed_dict)
        if global_step % 100 == 0 and not no_tf_log:
            self.train_writer.add_summary(summary, global_step)
        return
