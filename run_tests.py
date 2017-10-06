import os
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

DEFAULT_NUM_RUNS = 15
RUN_FILE = "framework.py"
LOG_DIR_ROOT = "logfiles"
agents = ["Qlearner",
          "KBQlearner",
          "CBQlearner"]
env = "CartPole-v1"
DEFAULT_NUM_ROLLOUTS = 2000

commands = []

log_dir = "log_archive/2000_Q-KBQ-CBQ_CartPole-v1"
try:
    os.mkdir(log_dir)
except:
    pass
for i in range(DEFAULT_NUM_RUNS):
    for agent in agents:
        commands.append(
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
for i in range(DEFAULT_NUM_RUNS):
    for agent in agents:
        commands.append(
            "python {0} {1} {2} --log_dir_root={3} --num_rollouts={4} --random_cartpole".format(
                RUN_FILE,
                agent,
                env,
                log_dir,
                DEFAULT_NUM_ROLLOUTS))

pool = Pool(8) # two concurrent commands at a time
for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
    if returncode != 0:
       print("%d command failed: %d" % (i, returncode))
