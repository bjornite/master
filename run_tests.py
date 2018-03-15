import os
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
from utilities import get_time_string, get_log_dir, parse_time_string


RUN_FILE = "framework.py"
LOG_DIR_ROOT = "experiments"

num_workers = 8
num_runs = 10
agents = [
    "Thompson",
    "BootDQN",
    "KBBoot"]
learning_rates = [1e-3, 5e-3]
epsilon = [10000, 1000]
experiments = [#("CartPole-v0", 600, [8, 8], "smalltwolayernet"),
               #("CartPole-v0", 600, [8, 8, 8], "smallthreelayernet"),
               #("CartPole-v0", 600, [32], "largeonelayernet"),
               #("CartPole-v0", 600, [32, 32], "largetwolayernet"),
               #("CartPole-v0", 600, [32, 32, 32], "largethreelayernet"),
               #("MountainCar-v0", 1500, [8], "smallonelayernet"),
               #("MountainCar-v0", 1500, [8, 8], "smalltwolayernet"),
    ("MountainCar-v0", 1500, [32], "largeonelayernet"),
    ("MountainCar-v0", 1500, [32, 32], "largetwolayernet"),
    ("MountainCar-v0", 1500, [8, 8, 8], "smallthreelayernet"),
    ("MountainCar-v0", 1500, [32, 32, 32], "largethreelayernet")]

log_tf = "--no_tf_log"
start_time = get_time_string()
start_datetime = parse_time_string(start_time)

commands = []

try:
    os.mkdir(LOG_DIR_ROOT)
except:
    pass
for i in range(len(experiments)):
    env, rollouts, hiddens, ldir = experiments[i]
    log_dir = LOG_DIR_ROOT + "/" + ldir
    for agent in agents:        
        for i in range(num_runs):
            for lr in learning_rates:
                for eps in epsilon:
                    if agent == "Thompson" and lr == 1e-3 and ldir == "largeonelayernet":
                        continue
                    commands.append(
                        "python {0} {1} {2} --log_dir_root={3} --num_rollouts={4} {5} --learning_rate {6} --n_hiddens {7} --epsilon {8}".format(
                            RUN_FILE,
                            agent,
                            env,
                            log_dir,
                            rollouts,
                            log_tf,
                            lr,
                            " ".join(str(x) for x in hiddens),
                            eps))

pool = Pool(num_workers)  # two concurrent commands at a time
for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
    if returncode != 0:
        print("%d command failed: %d" % (i, returncode))
