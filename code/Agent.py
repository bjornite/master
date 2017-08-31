class Agent(object):

    def train(self):
        raise NotImplementedError

    def __init__(self, name, env):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.name = name

    def get_action(self, input):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
