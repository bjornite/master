import os

DEFAULT_NUM_RUNS = 15
RUN_FILE = "framework.py"
LOG_DIR_ROOT = "logfiles"
agents = ["Qlearner",
          "KBQlearner",
          "CBQlearner"]
env = "CartPole-v1"
DEFAULT_NUM_ROLLOUTS = 2000

log_dir = "log_archive/2000_Q-KBQ-CBQ_CartPole-v1"
try:
    os.mkdir(log_dir)
except:
    pass
for agent in agents:
    for i in range(DEFAULT_NUM_RUNS):
        os.system(
            "python {0} {1} {2} --log_dir_root={3} --num_rollouts={4}".format(
                RUN_FILE,
                agent,
                env,
                log_dir,
                DEFAULT_NUM_ROLLOUTS))

log_dir = "log_archive/2000_Q-KBQ-CBQ_CartPole-v1-random"
try:
    os.mkdir(log_dir)
except:
    pass
for agent in agents:
    for i in range(DEFAULT_NUM_RUNS):
        os.system(
            "python {0} {1} {2} --log_dir_root={3} --num_rollouts={4} --random_cartpole".format(
                RUN_FILE,
                agent,
                env,
                log_dir,
                DEFAULT_NUM_ROLLOUTS))
