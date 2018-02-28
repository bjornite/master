from Agent import Agent
from tf_neural_net import CBTfTwoLayerNet
from tf_utils import LinearSchedule
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import numpy as np
import random
import copy
import os
import tensorflow as tf
import pickle

class DDQN(Agent):
    def __init__(self, name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon):
        super(DDQN, self).__init__(name, env, log_dir)

        inputsize = 0
        try:
            inputsize = self.observation_space.shape[0]
        except(AttributeError):
            inputsize = self.observation_space.n
        self.model = CBTfTwoLayerNet(inputsize,
                                     self.action_space.n,
                                     n_hiddens,
                                     learning_rate,
                                     reg_beta,
                                     self.log_dir)
        self.epsilon_schedule = LinearSchedule(1.0, epsilon, 0.02)
        self.sliding_target_updates = False
        self.prioritized_replay = False
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
        self.replay_memory_size = 1000000
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_eps = 1e-6
        if self.prioritized_replay:
            self.replay_memory = PrioritizedReplayBuffer(self.replay_memory_size,
                                                         self.prioritized_replay_alpha)
        else:
            self.replay_memory = ReplayBuffer(self.replay_memory_size)
        self.minibatch_size = 32
        self.old_weights = self.model.get_weights()
        self.target_update_freq = 500
        self.max_knowledge_reward = 0
        self.max_competence_reward = 0
        self.improvement_threshold = 0.2
        self.moving_average_uncertainties = 0
        self.protolog = []
        self.proto = [0, 0.1, 0, 0.1, 1]
        self.crlog = []
        self.bad = 0

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

    def make_priorities(self, td_errors, kb_rew, cb_rew):
        return np.abs(td_errors) + self.prioritized_replay_eps

    def train(self, no_tf_log):
        if self.replay_memory.length() < self.minibatch_size:
            return
        if self.prioritized_replay:
            states, a, obs, r, done, weights, batch_idxes = self.replay_memory.sample(self.minibatch_size, 0.5)
        else:
            states, a, obs, r, done = self.replay_memory.sample(self.minibatch_size)

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
        competence_rewards = self.model.get_meta_prediction_error(states,
                                                                  targetActionMask,
                                                                  knowledge_rewards,
                                                                  obs)
        c = 0
        #for i in range(len(states)):
        #    if np.sum(np.square(np.subtract(states[i], self.proto[:4]))) < 1e-1 and a[i] == self.proto[4]#:
 #               if len(self.protolog) > 0 and knowledge_rewards[i] > self.protolog[-1]:
  #                  self.bad += 1.0
  #              self.protolog.append(knowledge_rewards[i])
  #              self.crlog.append(competence_rewards[i])
  #              c += 1
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
        self.random_action_prob = self.epsilon_schedule.eps(self.training_steps)
        self.training_steps += 1
        td_errors = self.model.train(states,
                                     knowledge_rewards,
                                     obs,
                                     targets,
                                     targetActionMask,
                                     normalization_factor,
                                     no_tf_log)
        if self.prioritized_replay:
            new_priorities = self.make_priorities(td_errors, knowledge_rewards, competence_rewards)
            self.replay_memory.update_priorities(batch_idxes, new_priorities)
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

    def remember(self, s, a, r, stp1, d):
        self.replay_memory.add(s, a, r, stp1, d)

    def save_model(self, log_dir, filename):
        saver = tf.train.Saver()
        saver.save(self.model.sess, log_dir + "/" + filename)

    def load_model(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.model.sess, filename)

    def plot_state_visits(self):
        import matplotlib.pyplot as plt
        import latexipy as lp
        lp.latexify()  # Change to a serif font that fits with most LaTeX.
        if self.protolog != []:
            print("drawing error/mean/variance")
            # Calculate moving average
            avgs = [np.mean(self.protolog[i:i+10]) for i in range(len(self.protolog) - 10)]
            # Calculate local variance
            var = [np.std(self.protolog[i:i+10]) for i in range(len(self.protolog) - 10)]
            if len(self.protolog) > 100:
                with lp.figure("img1", size = lp.figure_size(n_columns=1)):
                    fig, ax1 = plt.subplots()
                    plotlen = 300
                    err, = ax1.plot(range(plotlen),
                                    self.protolog[:plotlen],
                                    label="Prediction error")
                    mean, = ax1.plot(range(plotlen),
                                     avgs[:plotlen],
                                     label="Moving average")
                    std, = ax1.plot(range(plotlen), var[:plotlen], label="Moving standard deviation")
                    #ax2 = ax1.twinx()
                    cr, = ax1.plot(range(plotlen), self.crlog[:plotlen], 'r', label="Error prediction")
                    plt.xlabel("Number of times state has been seen")
                    plt.ylabel("Error magnitude")
                    plt.legend(handles=[err, mean, std, cr])
            with lp.figure("img2", size = lp.figure_size(n_columns=1)):
                fig, ax1 = plt.subplots()
                err, = ax1.plot(range(len(self.protolog)),
                                self.protolog,
                                label="Prediction error")
                mean, = ax1.plot(range(len(avgs)),
                                 avgs,
                                 label="Moving average")
                std, = ax1.plot(range(len(var)), var, label="Moving standard deviation")
                #ax2 = ax1.twinx()
                cr, = ax1.plot(range(len(self.crlog)), self.crlog, 'r', label="Error prediction")
                plt.xlabel("Number of times state has been seen")
                plt.ylabel("Error magnitude")
                plt.legend(handles=[err, mean, std, cr])
            with lp.figure("img3", size=lp.figure_size(n_columns=1)):
                fig, ax1 = plt.subplots()
                err, = ax1.plot(range(len(self.protolog)-100, len(self.protolog)), self.protolog[-110:-10],
                                label="Prediction error")
                mean, = ax1.plot(range(len(self.protolog)-100, len(self.protolog)), avgs[-105:-5],
                                 label="Moving average")
                std, = ax1.plot(range(len(self.protolog)-100, len(self.protolog)), var[-105:-5],
                                label="Moving standard deviation")
                #ax2 = ax1.twinx()
                cr, = ax1.plot(range(len(self.crlog)-100, len(self.crlog)), self.crlog[-105:-5], 'r', label="Error prediction")
                plt.xlabel("Number of times state has been seen")
                plt.ylabel("Error magnitude")
                plt.legend(handles=[err, mean, std, cr])
            with open("{0}/protolog.pkl".format(self.log_dir), 'wb+') as f:
                pickle.dump(self.protolog, f)
            print(self.bad/(self.training_steps * self.minibatch_size))

    def debug_string(self):
        return ""


class KB(DDQN):

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = super(KB, self).make_reward(r,
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


class IKBQlearner(DDQN):

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = super(IKBQlearner, self).make_reward(r,
                                                       done,
                                                       max_q_values,
                                                       knowledge_rewards,
                                                       competence_rewards)
        max_knowledge_reward = np.max(knowledge_rewards)
        knowledge_rewards = [kr/max_knowledge_reward for kr in knowledge_rewards]
        for i in range(self.minibatch_size) and not done[i]:
            targets[i] -= knowledge_rewards[i]
        return targets


class CB(DDQN):

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = super(CB, self).make_reward(r,
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


class R(DDQN):

    def make_reward(self,
                    r,
                    done,
                    max_q_values,
                    #base_q_values,
                    knowledge_rewards,
                    competence_rewards):
        targets = super(R, self).make_reward(r,
                                                      done,
                                                      max_q_values,
                                                      knowledge_rewards,
                                                      competence_rewards)
        for i in range(self.minibatch_size):
            if not done[i]:
                targets[i] += random.random()
        return targets


class SAQlearner(DDQN):

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


class ISAQlearner(DDQN):

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


class MSAQlearner(DDQN):

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


class IMSAQlearner(DDQN):

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


class TESTQlearner(DDQN):

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
        if np.max(np.abs(cp_comprew)) > self.max_competence_reward:
            self.max_competence_reward = np.max(np.abs(cp_comprew))
        normalized_competence_rewards = [cr/self.max_competence_reward for cr in competence_rewards]
        rews = np.multiply(knowledge_rewards, normalized_competence_rewards)
        #avg = np.average(np.abs(normalized_competence_rewards))
        #normalized_competence_rewards = np.multiply(normalized_competence_rewards,
        #                                            0.5/avg)
        for i in range(self.minibatch_size):
            if competence_rewards[i] > 0 and not done[i]:
                targets[i] += rews[i]
        return targets


class Random_agent(Agent):
    def get_action(self, input, is_test=False):
        return self.action_space.sample()
