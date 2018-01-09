from Agent import Agent
from tf_neural_net import CBTfTwoLayerNet
from tf_utils import LinearSchedule
import numpy as np
import random
import copy
import os
import tensorflow as tf

class Qlearner(Agent):
    def __init__(self, name, env, log_dir, learning_rate, reg_beta):
        super(Qlearner, self).__init__(name, env, log_dir)

        inputsize = 0
        try:
            inputsize = self.observation_space.shape[0]
        except(AttributeError):
            inputsize = self.observation_space.n
        self.model = CBTfTwoLayerNet(inputsize,
                                     self.action_space.n,
                                     learning_rate,
                                     reg_beta,
                                     self.log_dir)
        self.epsilon_schedule = LinearSchedule(1.0, 100, 0.02)
        self.training_steps = 0
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
        self.old_weights = self.model.get_weights()
        self.target_update_freq = 500
        self.max_knowledge_reward = 0
        self.max_competence_reward = 0
        self.improvement_threshold = 0.2
        self.moving_average_uncertainties = 0

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    # base_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = np.zeros(self.minibatch_size)
        for i in range(self.minibatch_size):
            targets[i] = r[i]  # - base_q_values[i]
            if not done[i]:
                targets[i] += self.gamma*max_q_values[i]
        return targets

    def train(self, no_tf_log):
        data = random.sample(self.replay_memory, self.minibatch_size)
        states = [m[0] for m in data]
        a = [m[1] for m in data]
        obs = [m[2] for m in data]
        r = [m[3] for m in data]
        done = [m[4] for m in data]
        targetActionMask = np.zeros(
                        (self.minibatch_size, self.action_space.n), dtype=int)
        for i in range(self.minibatch_size):
            targetActionMask[i][a[i]] = 1
        target_actions = self.model.predict(obs)  # Double Q-learning
        target_q_values = self.model.predict(obs, weights=self.old_weights)

        max_q_values = [target_q_values[i][np.argmax(target_actions[i])]
                        for i in range(self.minibatch_size)]

        knowledge_rewards = self.model.get_prediction_error(states,
                                                            targetActionMask,
                                                            obs)
        max_knowledge_reward = np.max(knowledge_rewards)
        if max_knowledge_reward > self.max_knowledge_reward:
            self.max_knowledge_reward = max_knowledge_reward
        normalized_knowledge_rewards = [kr/self.max_knowledge_reward for kr in knowledge_rewards]
        competence_rewards = self.model.get_meta_prediction_error(states,
                                                                  targetActionMask,
                                                                  normalized_knowledge_rewards,
                                                                  obs)
        targets = self.make_reward(r,
                                   done,
                                   max_q_values,
                                   # base_q_values,
                                   knowledge_rewards,
                                   competence_rewards)
        for target in targets:
            self.target_mean = self.target_mean * (1-self.norm_beta) + self.norm_beta*target
            self.v = self.v * (1-self.norm_beta) + self.norm_beta*target*target
            self.target_sigma = max(1e-5, self.v - (self.target_mean * self.target_mean))
        normalization_factor = 1/(self.target_sigma*self.target_sigma)
        self.random_action_prob = self.epsilon_schedule.eps(self.training_steps)
        self.training_steps += 1
        self.model.train(states,
                         normalized_knowledge_rewards,
                         obs,
                         targets,
                         targetActionMask,
                         normalization_factor,
                         no_tf_log)
        # Update target network parameters
        count = 0
        if self.sliding_target_updates:
            current_weights = self.model.get_weights()
            for w in self.old_weights:
                self.old_weights[count] = np.add(np.multiply(self.old_weights[count], 0.999),
                                                 np.multiply(current_weights[count], (1-0.999)))
                if self.old_weights[count].shape[0] == 1:
                    self.old_weights[count] = self.old_weights[count].reshape([-1])
                count += 1
        elif self.training_steps % self.target_update_freq == 0:
            current_weights = self.model.get_weights()
            for w in self.old_weights:
                self.old_weights[count] = np.array(current_weights[count])
                if self.old_weights[count].shape[0] == 1:
                    self.old_weights[count] = self.old_weights[count].reshape([-1])
                count += 1

    def get_action(self, observation, is_test=False):
        if random.random() < self.random_action_prob and not is_test:
            return self.action_space.sample()
        else:
            values = self.model.predict([observation])
            return np.argmax(values[0])

    def remember(self, sars):
        self.replay_memory.append(sars)
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

    def save_model(self, log_dir, filename):
        saver = tf.train.Saver()
        saver.save(self.model.sess, log_dir + "/" + filename)

    def load_model(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.model.sess, filename)



class KBQlearner(Qlearner):

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    #base_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = super(KBQlearner, self).make_reward(r,
                                                      done,
                                                      max_q_values,
                                                      #base_q_values,
                                                      knowledge_rewards,
                                                      competence_rewards)
        max_knowledge_reward = np.max(knowledge_rewards)
        knowledge_rewards = [kr/max_knowledge_reward for kr in knowledge_rewards]
        for i in range(self.minibatch_size):
            targets[i] += 10 * knowledge_rewards[i]
        return targets

class IKBQlearner(Qlearner):

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    #base_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = super(IKBQlearner, self).make_reward(r,
                                                      done,
                                                      max_q_values,
                                                      #base_q_values,
                                                      knowledge_rewards,
                                                      competence_rewards)
        max_knowledge_reward = np.max(knowledge_rewards)
        knowledge_rewards = [kr/max_knowledge_reward for kr in knowledge_rewards]
        for i in range(self.minibatch_size):
            targets[i] -= knowledge_rewards[i]
        return targets

class CBQlearner(Qlearner):

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    #base_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = super(CBQlearner, self).make_reward(r,
                                                      done,
                                                      max_q_values,
                                                      #base_q_values,
                                                      knowledge_rewards,
                                                      competence_rewards)
        cp_comprew = copy.copy(competence_rewards)
        max_competence_reward = np.max(np.abs(cp_comprew))
        competence_rewards = [cr/max_competence_reward for cr in competence_rewards]
        for i in range(self.minibatch_size):
            if competence_rewards[i] > 0 and not done[i]:
                targets[i] += competence_rewards[i]
        return targets

class SAQlearner(Qlearner):

    def get_action(self, observation, is_test=False):
        if random.random() < self.random_action_prob and not is_test:
            # Return action causing maximum uncertainty
            uncertainties = np.zeros(self.action_space.n)
            for a in range(self.action_space.n):
                uncertainties[a] = self.model.get_prediction_uncertainty(observation, a)
            return np.argmax(uncertainties)
        else:
            values = self.model.predict([observation])
            return np.argmax(values[0])

class ISAQlearner(Qlearner):

    def get_action(self, observation, is_test=False):
        if random.random() < self.random_action_prob and not is_test:
            # Return action causing minimum uncertainty
            uncertainties = np.zeros(self.action_space.n)
            for a in range(self.action_space.n):
                uncertainties[a] = self.model.get_prediction_uncertainty(observation, a)
            return np.argmin(uncertainties)
        else:
            values = self.model.predict([observation])
            return np.argmax(values[0])

class MSAQlearner(Qlearner):

    def get_action(self, observation, is_test=False):
        if random.random() < self.random_action_prob and not is_test:
            # Return action causing average uncertainty
            uncertainties = np.zeros(self.action_space.n)
            for a in range(self.action_space.n):
                uncertainties[a] = self.model.get_prediction_uncertainty(observation, a)
            self.moving_average_uncertainties = (0.99 * self.moving_average_uncertainties
            + (1-0.99) * np.mean(uncertainties))
            uncertainties = np.abs(uncertainties - [self.moving_average_uncertainties])
            return np.argmin(uncertainties)
        else:
            values = self.model.predict([observation])
            return np.argmax(values[0])

class IMSAQlearner(Qlearner):

    def get_action(self, observation, is_test=False):
        if random.random() < self.random_action_prob and not is_test:
            # Return action causing average uncertainty
            uncertainties = np.zeros(self.action_space.n)
            for a in range(self.action_space.n):
                uncertainties[a] = self.model.get_prediction_uncertainty(observation, a)
            self.moving_average_uncertainties = (0.99 * self.moving_average_uncertainties
            + (1-0.99) * np.mean(uncertainties))
            uncertainties = np.abs(uncertainties - [self.moving_average_uncertainties])
            return np.argmax(uncertainties)
        else:
            values = self.model.predict([observation])
            return np.argmax(values[0])

class TESTQlearner(Qlearner):

    def get_action(self, observation, is_test=False):
        # Return action causing average uncertainty
        if random.random() < self.random_action_prob and not is_test:
            return self.action_space.sample()
        else:
            q_values = self.model.predict([observation])
            uncertainties = self.model.get_q_value_uncertainty([observation])
            max_estimate = q_values + uncertainties * self.random_action_prob
            return np.argmax(max_estimate)

class Random_agent(Agent):
    def get_action(self, input, is_test=False):
        return self.action_space.sample()
