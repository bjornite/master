import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


logger = logging.getLogger(__name__)


class DeepExploreBenchmark(gym.Env):

    def __init__(self):
        self.N = 10
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.N)

        self._seed()
        self.viewer = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        if action == 0:
            action = -1
        self.state = max(1, min(self.N, state + action))
        done = False

        if self.state == 1:
            reward = 1.0/1000
        elif self.state == self.N:
            reward = 1.0
        else:
            reward = 0.0
        self.obs = np.zeros(self.N)
        self.obs[:self.state+1] = 1
        return self.obs, reward, done, {}

    def _reset(self):
        self.state = 1
        self.obs = np.zeros(self.N)
        self.obs[:self.state+1] = 1
        self.steps_beyond_done = None
        return self.obs

    def _render(self, mode='human', close=False):
        pass
