from Agent import Agent
from tf_neural_net import TfTwoLayerNet
import numpy as np
import random


class Qlearner(Agent):
    def __init__(self, name, env):
        super(Qlearner, self).__init__(name, env)
        print(env.reward_range)
        self.model = TfTwoLayerNet(self.observation_space.shape[0],
                                   self.action_space.n)
        self.gamma = 0.99
        self.observations = []
        self.actions = []

    def train(self, obs, a, r):
        targets = []
        targetActionMask = []
        for i in range(1, len(obs)):
            obs_t = obs[i].tolist()
            old_weights = self.model.get_weights()
            Qnext = self.model.predict(np.asarray([obs_t]), weights=old_weights)[0][0]
            targets.append(r[i] + self.gamma*np.max(Qnext))
            targetActionMask.append([a[i] == j for j in range(self.action_space.n)])
        self.model.train(np.asarray(obs[1:]).reshape(-1, self.observation_space.shape[0]),
                         np.asarray(targets),
                         np.asarray(targetActionMask))

    def get_action(self, observation):
        values = self.model.predict(np.asarray(observation))
        if random.random() > 0.2:
            return np.argmax(values)
        else:
            return self.action_space.sample()

class Random_agent(Agent):
    def get_action(self, input):
        return self.action_space.sample()
