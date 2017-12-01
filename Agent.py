class Agent(object):

    def train(self):
        raise NotImplementedError

    def __init__(self, name, env, log_dir):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.name = name
        self.log_dir = log_dir

    def get_action(self, input):
        raise NotImplementedError

    def save_model(self, log_dir, filename):
        raise NotImplementedError
