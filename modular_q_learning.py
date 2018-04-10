from Agent import Agent
from tf_neural_net import CBTfTwoLayerNet, ModularNet
from tf_utils import LinearSchedule
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import numpy as np
import random
import copy
import os
import tensorflow as tf

class Module():
    def __init__(self, n_actions, args):
        self.net, self.predict, self.train_network, self.get_prediction_error, self.get_meta_prediction_error, self.get_weights, self.assign_weights = args
        self.sliding_target_updates = False
        self.prioritized_replay = False
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
        self.replay_memory_size = 1000000
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_eps = 1e-6
        if self.prioritized_replay:
            self.replay_memory = PrioritizedReplayBuffer(self.replay_memory_size,
                                                         self.prioritized_replay_alpha)
        else:
            self.replay_memory = ReplayBuffer(self.replay_memory_size)
        self.minibatch_size = 32
        self.old_weights = []
        self.target_update_freq = 500
        self.n_actions = n_actions

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = np.zeros(self.minibatch_size)
        for i in range(self.minibatch_size):
            targets[i] = r[i]
            if not done[i]:
                targets[i] += self.gamma*max_q_values[i]
        return targets

    def make_priorities(self, td_errors, kb_rew, cb_rew):
        return np.abs(td_errors) + self.prioritized_replay_eps

    def train(self, no_tf_log):
        if self.replay_memory.length() < self.minibatch_size:
            return
        if self.prioritized_replay:
            states, a, obs, r, done, weights, batch_idxes = self.replay_memory.sample(self.minibatch_size, 0.5)
        else:
            states, a, obs, r, done = self.replay_memory.sample(self.minibatch_size)
        targetActionMask = np.zeros((self.minibatch_size, self.n_actions), dtype=int)
        for i in range(self.minibatch_size):
            targetActionMask[i][a[i]] = 1
        target_actions = self.predict(self.net, obs)  # Double Q-learning
        if self.old_weights == []:
            self.old_weights = self.get_weights(self.net)
        target_q_values = self.predict(self.net, obs, input_weights=self.old_weights)

        max_q_values = [target_q_values[i][np.argmax(target_actions[i])]
                        for i in range(self.minibatch_size)]

        knowledge_rewards = self.get_prediction_error(self.net,
                                                      states,
                                                      targetActionMask,
                                                      obs)
        competence_rewards = self.get_meta_prediction_error(self.net,
                                                            states,
                                                            targetActionMask,
                                                            knowledge_rewards,
                                                            obs)
        targets = self.make_reward(r,
                                   done,
                                   max_q_values,
                                   knowledge_rewards,
                                   competence_rewards)
        for target in targets:
            self.target_mean = self.target_mean * (1-self.norm_beta) + self.norm_beta*target
            self.v = self.v * (1-self.norm_beta) + self.norm_beta*target*target
            self.target_sigma = max(1e-5, self.v - (self.target_mean * self.target_mean))
        normalization_factor = 1/(self.target_sigma*self.target_sigma)
        self.training_steps += 1
        self.global_steps += 1
        td_errors = self.train_network(self.net,
                                       states,
                                       knowledge_rewards,
                                       obs,
                                       targets,
                                       targetActionMask,
                                       normalization_factor,
                                       no_tf_log)
        if self.prioritized_replay:
            new_priorities = self.make_priorities(td_errors, knowledge_rewards, competence_rewards)
            self.replay_memory.update_priorities(batch_idxes, new_priorities)
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
        values = self.predict(self.net, [observation])
        return np.argmax(values[0])

    def remember(self, s, a, r, stp1, d):
        self.replay_memory.add(s, a, r, stp1, d)

    def weights(self):
        return self.get_weights(self.net)

    def set_weights(self, weights):
        self.assign_weights(self.net, weights)


class BootDQN(Agent):
    def __init__(self, name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon):
        super(BootDQN, self).__init__(name, env, log_dir)
        inputsize = 0
        try:
            inputsize = self.observation_space.shape[0]
        except(AttributeError):
            inputsize = self.observation_space.n
        self.modules = []
        config = tf.ConfigProto(
            device_count={'GPU': 0}
            )
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.sess = tf.Session(config=config)
        self.model = ModularNet(inputsize,
                                self.action_space.n,
                                n_hiddens,
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
        self.active_module = 0
        self.activated_modules = 5
        self.max_number_of_modules = 5
        self.action_steps = 0.0
        self.active_module_counter = []
        for i in range(self.max_number_of_modules):
            self.make_new_module()
        self.model.init()

    def make_new_module(self):
        new_module = Module(self.action_space.n, self.model.make_module(len(self.modules)))
        self.active_module_counter.append(0)
        self.modules.append(new_module)

    def get_action(self, observation, is_test=False):
        self.action_steps += 1.0
        self.active_module_counter[self.active_module] += 1
        return self.modules[self.active_module].get_action(observation, is_test)

    def remember(self, s, a, r, stp1, d):
        if d:
            self.active_module = random.randint(0, self.activated_modules-1)
        for m in self.modules:
            if random.random() < 0.05:
                continue
            else:
                m.remember(s, a, r, stp1, d)

    def train(self, no_tf_log):
        self.training_steps += 1
        for m in self.modules:
            m.train(no_tf_log)

    def debug_string(self):
        return "Number of modules: {0} Module use: {1}".format(
            self.activated_modules, str(self.active_module_counter))

class EpsBootDQN(BootDQN):
    
    def get_action(self, observation, is_test=False):
        if random.random() < 0.02:
            return self.action_space.sample()
        else:
            self.action_steps += 1.0
            self.active_module_counter[self.active_module] += 1
            return self.modules[self.active_module].get_action(observation, is_test)

class KBModule(Module):

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = super(KBModule, self).make_reward(r,
                                                    done,
                                                    max_q_values,
                                                    knowledge_rewards,
                                                    competence_rewards)
        max_knowledge_reward = np.max(knowledge_rewards)
        normalized_knowledge_rewards = [kr/max_knowledge_reward for kr in knowledge_rewards]
        for i in range(self.minibatch_size):
            if not done[i]:
                targets[i] += normalized_knowledge_rewards[i]
        return targets


class CBModule(Module):

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = super(CBModule, self).make_reward(r,
                                                    done,
                                                    max_q_values,
                                                    knowledge_rewards,
                                                    competence_rewards)
        cp_comprew = copy.copy(competence_rewards)
        max_competence_reward = np.max(np.abs(cp_comprew))
        normalized_competence_rewards = [cr/max_competence_reward for cr in competence_rewards]
        for i in range(self.minibatch_size):
            if competence_rewards[i] > 0 and not done[i]:
                targets[i] += normalized_competence_rewards[i]
        return targets


class KBBoot(BootDQN):

    def make_new_module(self):
        new_module = KBModule(self.action_space.n, self.model.make_module(len(self.modules)))
        self.active_module_counter.append(0)
        self.modules.append(new_module)


class CBBoot(BootDQN):

    def make_new_module(self):
        new_module = CBModule(self.action_space.n, self.model.make_module(len(self.modules)))
        self.active_module_counter.append(0)
        self.modules.append(new_module)


class Thompson(BootDQN):

    def remember(self, s, a, r, stp1, d):
        self.active_module = random.randint(0, self.activated_modules-1)
        for m in self.modules:
            if random.random() < 0.05:
                continue
            else:
                m.remember(s, a, r, stp1, d)


class AllCombined(BootDQN):

    def make_new_module(self):
        new_module = KBModule(self.action_space.n, self.model.make_module(len(self.modules)))
        self.active_module_counter.append(0)
        self.modules.append(new_module)

    def get_action(self, observation, is_test=False):
        self.action_steps += 1.0
        if random.random() < 0.05:
            random_module = random.randint(0,self.activated_modules-1)
            return self.modules[random_module].get_action(observation, is_test)
        else:
            self.active_module_counter[self.active_module] += 1
            return self.modules[self.active_module].get_action(observation, is_test)
