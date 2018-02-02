from Agent import Agent
from tf_neural_net import CBTfTwoLayerNet, ModularNet
from tf_utils import LinearSchedule
import numpy as np
import random
import copy
import os
import tensorflow as tf

class Module():
    def __init__(self, n_actions, args):
        self.net, self.predict, self.train_network, self.get_prediction_error, self.get_weights, self.assign_weights = args
        self.epsilon_schedule = LinearSchedule(1.0, 100, 0.02)
        self.sliding_target_updates = False
        self.training_steps = 0
        self.global_steps = 0
        self.gamma = 1.0
        self.tau = 0.01
        self.norm_beta = 0.1
        self.random_action_prob = 1
        self.target_mean = 0
        self.target_sigma = 0
        self.v = 0
        self.observations = []
        self.actions = []
        self.replay_memory = []
        self.replay_memory_size = 50000
        self.minibatch_size = 32
        self.old_weights = []
        self.target_update_freq = 500
        self.max_knowledge_reward = 0
        self.max_competence_reward = 0
        self.improvement_threshold = 0.2
        self.moving_average_uncertainties = 0
        self.avg_pred_error = 0
        self.local_avg_pred_error = 0
        self.k = 1e-1
        self.progress = 10
        self.progress_increment = 0.1
        self.progress_decay_factor = 0.9
        self.times_used = 0
        self.n_actions = n_actions

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    # base_q_values,
                    knowledge_rewards):
        targets = np.zeros(self.minibatch_size)
        for i in range(self.minibatch_size):
            targets[i] = r[i]  # - base_q_values[i]
            if not done[i]:
                targets[i] += self.gamma*max_q_values[i]
        return targets

    #def make_reward(self,
    #                r,
    #                done,
    #                max_q_values,
    #                # base_q_values,
    #                knowledge_rewards):
    #    targets = self.make_q_reward(r,
    #                                 done,
    #                                 max_q_values,
    #                                 # base_q_values,
    #                                 knowledge_rewards)
    #    max_knowledge_reward = np.max(knowledge_rewards)
    #    knowledge_rewards = [kr/max_knowledge_reward for kr in knowledge_rewards]
    #    for i in range(self.minibatch_size):
    #        targets[i] += knowledge_rewards[i]
    #    return targets

    def train(self, no_tf_log):
        if len(self.replay_memory) < self.minibatch_size:
            return
        data = random.sample(self.replay_memory, self.minibatch_size)
        states = [m[0] for m in data]
        a = [m[1] for m in data]
        obs = [m[2] for m in data]
        r = [m[3] for m in data]
        done = [m[4] for m in data]
        targetActionMask = np.zeros(
                        (self.minibatch_size, self.n_actions), dtype=int)
        for i in range(self.minibatch_size):
            targetActionMask[i][a[i]] = 1
        target_actions = self.predict(self.net, obs)  # Double Q-learning
        if self.old_weights == []:
            print("Updated old weights")
            self.old_weights = self.get_weights(self.net)
        target_q_values = self.predict(self.net, obs, input_weights=self.old_weights)

        max_q_values = [target_q_values[i][np.argmax(target_actions[i])]
                        for i in range(self.minibatch_size)]

        knowledge_rewards = self.get_prediction_error(self.net,
                                                      states,
                                                      targetActionMask,
                                                      obs)
        max_knowledge_reward = np.max(knowledge_rewards)
        if max_knowledge_reward > self.max_knowledge_reward:
            self.max_knowledge_reward = max_knowledge_reward
        normalized_knowledge_rewards = [kr/self.max_knowledge_reward for kr in knowledge_rewards]

        targets = self.make_reward(r,
                                   done,
                                   max_q_values,
                                   # base_q_values,
                                   knowledge_rewards)
        for target in targets:
            self.target_mean = self.target_mean * (1-self.norm_beta) + self.norm_beta*target
            self.v = self.v * (1-self.norm_beta) + self.norm_beta*target*target
            self.target_sigma = max(1e-5, self.v - (self.target_mean * self.target_mean))
        normalization_factor = 1/(self.target_sigma*self.target_sigma)
        self.random_action_prob = self.epsilon_schedule.eps(self.global_steps)
        self.training_steps += 1
        self.global_steps += 1
        self.train_network(self.net,
                           states,
                           normalized_knowledge_rewards,
                           obs,
                           targets,
                           targetActionMask,
                           normalization_factor,
                           no_tf_log)
        #Update target network parameters
        count = 0
        if self.sliding_target_updates:
            current_weights = self.get_weights(self.net)
            for w in self.old_weights:
                self.old_weights[count] = np.add(np.multiply(self.old_weights[count], 0.999),
                                                  np.multiply(current_weights[count], (1-0.999)))
                if self.old_weights[count].shape[0] == 1:
                    self.old_weights[count] = self.old_weights[count].reshape([-1])
                count += 1
        elif self.global_steps % self.target_update_freq == 0:
            current_weights = self.get_weights(self.net)
            for w in self.old_weights:
                self.old_weights[count] = np.array(current_weights[count])
                if self.old_weights[count].shape[0] == 1:
                    self.old_weights[count] = self.old_weights[count].reshape([-1])
                count += 1

    def get_action(self, observation, is_test=False):
        if random.random() < self.random_action_prob and not is_test:
            return np.random.randint(0, self.n_actions)
        else:
            values = self.predict(self.net, [observation])
        return np.argmax(values[0])

    def get_highest_q_value(self, observation):
        values = self.predict(self.net, [observation])
        return np.max(values[0])

    def remember(self, sars):
        states = [sars[0]]
        targetActionMask = np.zeros(self.n_actions)
        targetActionMask[sars[1]] = 1
        obs = [sars[2]]
        current_pred_error = self.get_prediction_error(self.net,
                                                       states,
                                                       [targetActionMask],
                                                       obs)
        # Update freq
        self.times_used += 1.0
        # Update progress
        self.progress *= self.progress_decay_factor
        if current_pred_error < self.local_avg_pred_error:
            self.progress += self.progress_increment
        # Update local moving average prediction error
        self.local_avg_pred_error = ((1 - self.k) *
                                     self.local_avg_pred_error +
                                     self.k *
                                     current_pred_error)
        self.replay_memory.append(sars)
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

    def get_avg_pred_error(self, observation):
        values = self.predict(self.net, [observation])
        if np.argmax(values[0]) == 0:
            return self.avg_pred_error
        else:
            return self.avg_pred_error / np.max(values[0])

    def get_progress(self):
        return self.progress

    def get_frequency(self, action_steps):
        if action_steps > 0:
            return self.times_used / action_steps
        else:
            return 0

    def weights(self):
        return self.get_weights(self.net)

    def set_weights(self, weights):
        self.assign_weights(self.net, weights)


class ModularDQN(Agent):
    def __init__(self, name, env, log_dir, learning_rate, reg_beta):
        super(ModularDQN, self).__init__(name, env, log_dir)
        inputsize = 0
        try:
            inputsize = self.observation_space.shape[0]
        except(AttributeError):
            inputsize = self.observation_space.n
        self.modules = []
        config = tf.ConfigProto(
            device_count={'GPU': 0}
            )
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)
        self.model = ModularNet(inputsize,
                                self.action_space.n,
                                learning_rate,
                                reg_beta,
                                self.log_dir,
                                self.sess)
        self.name = name
        self.env = env
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.reg_beta = reg_beta
        self.training_steps = 0
        self.random_action_prob = 1
        self.epsilon_schedule = LinearSchedule(1.0, 100, 0.02)
        self.active_module = 0
        self.activated_modules = 5
        self.max_number_of_modules = 5
        self.action_steps = 0.0
        self.active_module_counter = []
        for i in range(self.max_number_of_modules):
            self.modules.append(Module(self.action_space.n, self.model.make_module(i)))
            self.active_module_counter.append(0)
        self.model.init()

    def make_new_module(self, module):
        new_module = Module(self.action_space.n, self.model.make_module(len(self.modules)))
        #new_module.model.weights = module.model.get_weights()
        self.active_module_counter.append(0)
        self.modules.append(new_module)

    def get_action(self, observation, is_test=False):
        self.action_steps += 1.0
        # Get average prediction error of last frames from all modules
        #if random.random() < self.random_action_prob and not is_test:
        #    best_module_idx = np.random.randint(0, self.activated_modules)
        #else:
        pred_errors = []
        q_vals = []
        progression = []
        for i in range(self.activated_modules):
            pred_errors.append(float(self.modules[i].get_avg_pred_error(observation)))
            q_vals.append(float(self.modules[i].get_highest_q_value(observation)))
            progression.append(float(self.modules[i].get_progress()))
        #best_module_idx = np.argmin(pred_errors)
        #best_module_idx = np.argmax(q_vals)
        progression = np.multiply(progression, q_vals)
        #best_module_idx = np.argmax(q_vals)
        best_module_idx = self.active_module
        #if observation[0] > 0:
        #    best_module_idx = 0
        #else:
        #    best_module_idx = 1
        #print(best_module_idx)
        if (self.modules[best_module_idx].get_progress() < 0.5 and
            self.modules[best_module_idx].get_frequency(self.action_steps) > 1.0/self.activated_modules - 0.1 and
            self.activated_modules < self.max_number_of_modules and self.action_steps > 100):
            self.activated_modules += 1
            best_weights = self.modules[best_module_idx].weights()
            # Copy values from best module to newly activated module
            self.modules[self.activated_modules-1].set_weights(best_weights)
            self.modules[self.activated_modules-1].replay_memory = self.modules[best_module_idx].replay_memory
            self.modules[self.activated_modules-1].training_steps = self.modules[best_module_idx].training_steps
            self.modules[self.activated_modules-1].avg_pred_error = self.modules[best_module_idx].avg_pred_error
            # Reset frequency counters
            for m in self.modules:
                m.times_used = 0
            self.action_steps = 0
        self.active_module = best_module_idx
        self.active_module_counter[self.active_module] += 1
        return self.modules[best_module_idx].get_action(observation, is_test)

    def remember(self, sars):
        # update moving average prediction error for all modules
        states = [sars[0]]
        targetActionMask = np.zeros(self.action_space.n)
        targetActionMask[sars[1]] = 1
        obs = sars[2]
        for m in self.modules:
            m.avg_pred_error = ((1 - m.k) * m.avg_pred_error +
                                m.k * m.get_prediction_error(m.net,
                                                             states,
                                                             [targetActionMask],
                                                             [obs]))
            if sars[4]: # Reset on start of new episode
                m.avg_pred_error = 1
                self.active_module = np.random.randint(0, self.activated_modules)
        #if obs[0] < 0.2 and obs[0] > -0.2:
        #    self.modules[0].remember(sars)
        #    self.modules[1].remember(sars)
        #else:
        for m in self.modules:
            if random.random() < 0.05:
                continue
            else:
                m.remember(sars)
        # Save observation to active module
        #self.modules[self.active_module].remember(sars)

    def train(self, no_tf_log):
        self.random_action_prob = self.epsilon_schedule.eps(self.training_steps)
        self.training_steps += 1
        for m in self.modules:
            m.train(no_tf_log)
        #self.modules[self.active_module].train(no_tf_log)

    def debug_string(self):
        arr = []
        for m in self.modules:
            arr.append(float(m.get_progress()))
        return "Number of modules: {0} Module use: {1} Arr; {2}".format(self.activated_modules, str(self.active_module_counter), arr)
