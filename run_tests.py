import os
import argparse
import datetime
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
from utilities import get_time_string, get_log_dir, parse_time_string


DEFAULT_NUM_RUNS = 15
RUN_FILE = "framework.py"
LOG_DIR_ROOT = "logfiles"
agents = ["Qlearner",
          "KBQlearner",
          "CBQlearner"]
env = "CartPole-v1"
DEFAULT_NUM_ROLLOUTS = 2000
DEFAULT_NUM_WORKERS = 7

parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=DEFAULT_NUM_RUNS)
parser.add_argument('--envname', type=str, default=env)
parser.add_argument('--num_rollouts', type=int, default=DEFAULT_NUM_ROLLOUTS)
parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
args = parser.parse_args()

start_time = get_time_string()
start_datetime = parse_time_string(start_time)

commands = []

log_dir = "log_archive/{0}_Q-KBQ-CBQ_CartPole-v1_{1}".format(args.num_rollouts, start_time)
try:
    os.mkdir(log_dir)
except:
    pass
for i in range(args.num_runs):
    for agent in agents:
        commands.append(
            "python {0} {1} {2} --log_dir_root={3} --num_rollouts={4}".format(
                RUN_FILE,
                agent,
                env,
                log_dir,
                args.num_rollouts))

log_dir = "log_archive/{0}_Q-KBQ-CBQ_CartPole-v1-random_{1}".format(args.num_rollouts, start_time)
try:
    os.mkdir(log_dir)
except:
    pass
for i in range(args.num_runs):
    for agent in agents:
        commands.append(
            "python {0} {1} {2} --log_dir_root={3} --num_rollouts={4} --random_cartpole".format(
                RUN_FILE,
                agent,
                env,
                log_dir,
                args.num_rollouts))

pool = Pool(args.num_workers)  # two concurrent commands at a time
for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
    if returncode != 0:
        print("%d command failed: %d" % (i, returncode))
