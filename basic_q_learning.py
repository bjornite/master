from Agent import Agent
from tf_neural_net import CBTfTwoLayerNet
import numpy as np
import random
import copy
import os
import tensorflow as tf

class Qlearner(Agent):
    def __init__(self, name, env, log_dir, learning_rate, reg_beta):
        super(Qlearner, self).__init__(name, env, log_dir)
        self.model = CBTfTwoLayerNet(self.observation_space.shape[0],
                                     self.action_space.n,
                                     learning_rate,
                                     reg_beta,
                                     self.log_dir)
        self.gamma = 0.999
        self.tau = 0.01
        self.random_action_prob = 0.9
        self.random_action_decay = 0.99999
        self.observations = []
        self.actions = []
        self.replay_memory = []
        self.replay_memory_size = 1000000
        self.minibatch_size = 1000
        self.old_weights = self.model.get_weights()
        self.target_update_freq = 100
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
        #self.model.learning_rate *= 0.99995 # Converges in roughly 200000 steps
        self.random_action_prob *= self.random_action_decay
        self.model.train(states,
                         normalized_knowledge_rewards,
                         obs,
                         targets,
                         targetActionMask,
                         no_tf_log)

    def get_action(self, observation, is_test=False):
        if random.random() < self.random_action_prob and not is_test:
            return self.action_space.sample()
        else:
            values = self.model.predict([observation])
            return np.argmax(values[0])

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
            targets[i] += knowledge_rewards[i]
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
            if competence_rewards[i] > 0:
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
