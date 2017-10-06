from Agent import Agent
from tf_neural_net import TfTwoLayerNet, KBTfTwoLayerNet, CBTfTwoLayerNet
import numpy as np
import random


class Qlearner(Agent):
    def __init__(self, name, env, log_dir):
        super(Qlearner, self).__init__(name, env, log_dir)
        self.model = TfTwoLayerNet(self.observation_space.shape[0],
                                   self.action_space.n,
                                   self.log_dir)
        self.gamma = 0.9
        self.tau = 0.01
        self.random_action_prob = 0.5
        self.random_action_decay = 0.999
        self.observations = []
        self.actions = []
        self.replay_memory = []
        self.replay_memory_size = 50000
        self.minibatch_size = 100
        self.old_weights = self.model.get_weights()
        self.target_update_freq = 1000

    def train(self):
        data = random.sample(self.replay_memory, self.minibatch_size)
        states = [m[0] for m in data]
        a = [m[1] for m in data]
        obs = [m[2] for m in data]
        r = [m[3] for m in data]
        targets = np.zeros(len(data))
        targetActionMask = np.zeros((self.minibatch_size, self.action_space.n), dtype=int)
        target_actions = self.model.predict(obs)
        target_q_values = self.model.predict(obs, weights=self.old_weights)
        baseline_q_values = self.model.predict(states, weights=self.old_weights)
        max_q_values = [target_q_values[i][np.argmax(target_actions[i])] for i in range(len(target_q_values))]
        base_q_values = [baseline_q_values[i][np.argmax(target_actions[i])] for i in range(len(baseline_q_values))]
        for i in range(len(data)):
            targets[i] = r[i]
            if not data[i][4]:
                targets[i] += (self.gamma*max_q_values[i] - base_q_values[i])
            targetActionMask[i][a[i]] = 1
        self.model.train(states,
                         targets,
                         targetActionMask)

    def get_action(self, observation):
        self.random_action_prob *= self.random_action_decay
        values = self.model.predict([observation])
        if random.random() > self.random_action_prob:
            return np.argmax(values)
        else:
            return self.action_space.sample()

class KBQlearner(Agent):
    def __init__(self, name, env, log_dir):
        super(KBQlearner, self).__init__(name, env, log_dir)
        print(env.reward_range)
        self.model = KBTfTwoLayerNet(self.observation_space.shape[0],
                                     self.action_space.n,
                                     self.log_dir)
        self.gamma = 0.9
        self.tau = 0.01
        self.random_action_prob = 0.5
        self.random_action_decay = 0.999
        self.observations = []
        self.actions = []
        self.replay_memory = []
        self.replay_memory_size = 200000
        self.minibatch_size = 100
        self.old_weights = self.model.get_weights()
        self.target_update_freq = 1000

    def train(self):
        data = random.sample(self.replay_memory, self.minibatch_size)
        states = [m[0] for m in data]
        a = [m[1] for m in data]
        obs = [m[2] for m in data]
        r = [m[3] for m in data]
        targets = np.zeros(len(data))
        targetActionMask = np.zeros(
                        (self.minibatch_size, self.action_space.n), dtype=int)
        target_actions = self.model.predict(obs)
        target_q_values = self.model.predict(obs, weights=self.old_weights)
        baseline_q_values = self.model.predict(states, weights=self.old_weights)
        max_q_values = [target_q_values[i][np.argmax(target_actions[i])] for i in range(len(target_q_values))]
        base_q_values = [baseline_q_values[i][np.argmax(target_actions[i])] for i in range(len(baseline_q_values))]
        knowledge_rewards = self.model.get_prediction_error(states, obs)
        max_knowledge_reward = np.max(knowledge_rewards)
        knowledge_rewards = [kr/max_knowledge_reward for kr in knowledge_rewards]
        for i in range(len(data)):
            targets[i] = r[i]
            if not data[i][4]:
                targets[i] += (self.gamma*max_q_values[i] -
                               base_q_values[i])
                targets[i] += knowledge_rewards[i]
            targetActionMask[i][a[i]] = 1
        self.model.train(states,
                         obs,
                         targets,
                         targetActionMask)

    def get_action(self, observation):
        self.random_action_prob *= self.random_action_decay
        values = self.model.predict([observation])
        if random.random() > self.random_action_prob:
            return np.argmax(values)
        else:
            return self.action_space.sample()

class CBQlearner(Agent):
    def __init__(self, name, env, log_dir):
        super(CBQlearner, self).__init__(name, env, log_dir)
        print(env.reward_range)
        self.model = CBTfTwoLayerNet(self.observation_space.shape[0],
                                     self.action_space.n,
                                     self.log_dir)
        self.gamma = 0.9
        self.tau = 0.01
        self.random_action_prob = 0.5
        self.random_action_decay = 0.999
        self.observations = []
        self.actions = []
        self.replay_memory = []
        self.replay_memory_size = 200000
        self.minibatch_size = 100
        self.old_weights = self.model.get_weights()
        self.target_update_freq = 1000
        self.max_knowledge_reward = 0
        self.max_competence_reward = 0
        self.improvement_threshold = 0.2

    def train(self):
        data = random.sample(self.replay_memory, self.minibatch_size)
        states = [m[0] for m in data]
        a = [m[1] for m in data]
        obs = [m[2] for m in data]
        r = [m[3] for m in data]
        targets = np.zeros(len(data))
        targetActionMask = np.zeros(
                        (self.minibatch_size, self.action_space.n), dtype=int)
        target_actions = self.model.predict(obs)
        target_q_values = self.model.predict(obs, weights=self.old_weights)
        baseline_q_values = self.model.predict(states, weights=self.old_weights)
        max_q_values = [target_q_values[i][np.argmax(target_actions[i])] for i in range(len(target_q_values))]
        base_q_values = [baseline_q_values[i][np.argmax(target_actions[i])] for i in range(len(baseline_q_values))]
        knowledge_rewards = self.model.get_prediction_error(states,
                                                            obs)
        max_knowledge_reward = np.max(knowledge_rewards)
        if max_knowledge_reward > self.max_knowledge_reward:
            self.max_knowledge_reward = max_knowledge_reward
        knowledge_rewards = [kr/self.max_knowledge_reward for kr in knowledge_rewards]
        competence_rewards = self.model.get_meta_prediction_error(states,
                                                                  knowledge_rewards,
                                                                  obs)
        max_competence_reward = np.max(competence_rewards)
        competence_rewards = [cr/max_competence_reward for cr in competence_rewards]
        for i in range(len(data)):
            targets[i] = r[i]
            if not data[i][4]:
                targets[i] += (self.gamma*max_q_values[i] -
                               base_q_values[i])
                # targets[i] -= knowledge_rewards[i]
                if competence_rewards[i] > self.improvement_threshold and knowledge_rewards:
                    targets[i] += competence_rewards[i]
            targetActionMask[i][a[i]] = 1
        self.model.train(states,
                         knowledge_rewards,
                         obs,
                         targets,
                         targetActionMask)
        return np.sum([1.0/len(competence_rewards) for x in competence_rewards if x > self.improvement_threshold])

    def get_action(self, observation):
        self.random_action_prob *= self.random_action_decay
        values = self.model.predict([observation])
        if random.random() > self.random_action_prob:
            return np.argmax(values)
        else:
            return self.action_space.sample()

class Random_agent(Agent):
    def get_action(self, input):
        return self.action_space.sample()
