from Agent import Agent
from tf_neural_net import CBTfTwoLayerNet
import numpy as np
import random

class Qlearner(Agent):
    def __init__(self, name, env, log_dir, learning_rate, reg_beta):
        super(Qlearner, self).__init__(name, env, log_dir)
        self.model = CBTfTwoLayerNet(self.observation_space.shape[0]*2,
                                     self.action_space.n,
                                     learning_rate,
                                     reg_beta,
                                     self.log_dir)
        self.gamma = 0.999
        self.tau = 0.01
        self.random_action_prob = 0.9
        self.random_action_decay = 1.0
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
        # baseline_q_values = self.model.predict(states)
        max_q_values = [target_q_values[i][np.argmax(target_actions[i])]
                        for i in range(self.minibatch_size)]
        # base_q_values = [baseline_q_values[i][a[i]]
        #                 for i in range(len(baseline_q_values))]
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
        self.model.train(states,
                         normalized_knowledge_rewards,
                         obs,
                         targets,
                         targetActionMask,
                         no_tf_log)

    def get_action(self, observation, is_test=False):
        self.random_action_prob *= self.random_action_decay
        values = self.model.predict([observation])
        if random.random() < self.random_action_prob/4 and not is_test:
            return self.action_space.sample()
        else:
            return np.argmax(values[0])


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
        if max_knowledge_reward > 1:
            knowledge_rewards = [kr/max_knowledge_reward for kr in knowledge_rewards]
        for i in range(self.minibatch_size):
            targets[i] += knowledge_rewards[i]
        return targets

    def get_action(self, observation, is_test=False):
        values = self.model.predict([observation])
        if random.random() < self.random_action_prob/4 and not is_test:
            return self.action_space.sample()
        else:
            return np.argmax(values[0])

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
        if max_knowledge_reward > 1:
            knowledge_rewards = [kr/max_knowledge_reward for kr in knowledge_rewards]
        for i in range(self.minibatch_size):
            targets[i] -= knowledge_rewards[i]
        return targets

    def get_action(self, observation, is_test=False):
        values = self.model.predict([observation])
        if random.random() < self.random_action_prob/4 and not is_test:
            return self.action_space.sample()
        else:
            return np.argmax(values[0])

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
        max_competence_reward = np.max(competence_rewards)
        if max_competence_reward > 1:
            competence_rewards = [cr/max_competence_reward for cr in competence_rewards]
        for i in range(self.minibatch_size):
            if competence_rewards[i]:
                targets[i] += competence_rewards[i]
        return targets

    def get_action(self, observation, is_test=False):
        values = self.model.predict([observation])
        if random.random() < self.random_action_prob/4 and not is_test:
            return self.action_space.sample()
        else:
            return np.argmax(values[0])

class SAQlearner(Qlearner):

    def get_action(self, s):
        uncertainties = {}
        for a in self.action_space():
            uncertainties[a] = self.model.get_prediction_uncertainty(s, a)
        # Sort uncertainties
        self.random_action_prob *= self.random_action_decay
        values = self.model.predict([observation])
        if random.random() > self.random_action_prob:
            # Return action causing median uncertainty
            return np.argmax(values)
        else:
            return self.action_space.sample()

class Random_agent(Agent):
    def get_action(self, input):
        return self.action_space.sample()
