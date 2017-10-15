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
DEFAULT_AGENT = "Qlearner"
DEFAULT_ENV = "CartPole-v1"
DEFAULT_NUM_ROLLOUTS = 500
DEFAULT_NUM_WORKERS = 7
DEFAULT_LEARNING_RATE=1e-3

parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=DEFAULT_NUM_RUNS)
parser.add_argument('--agentname', type=str, default=DEFAULT_AGENT)
parser.add_argument('--envname', type=str, default=DEFAULT_ENV)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=DEFAULT_NUM_ROLLOUTS)
parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
parser.add_argument('--log_tf', action='store_true')
parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
args = parser.parse_args()

start_time = get_time_string()
start_datetime = parse_time_string(start_time)
log_tf = "--no_tf_log"
if args.log_tf:
    log_tf = ""

commands = []

for i in range(args.num_runs):
    commands.append(
        "python {0} {1} {2} --log_dir_root={3} --num_rollouts={4} {5} --learning_rate={6}".format(
            RUN_FILE,
            args.agentname,
            args.envname,
            LOG_DIR_ROOT,
            args.num_rollouts,
            log_tf,
            args.learning_rate))

pool = Pool(args.num_workers)  # two concurrent commands at a time
for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
    if returncode != 0:
        print("%d command failed: %d" % (i, returncode))

print("Success")
