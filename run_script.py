import os
import argparse
import datetime
from utilities import get_time_string, get_log_dir, parse_time_string

DEFAULT_NUM_RUNS = 15
RUN_FILE = "framework.py"
LOG_DIR_ROOT = "logfiles"
DEFAULT_AGENT = "Qlearner"
DEFAULT_ENV = "CartPole-v1"
DEFAULT_NUM_ROLLOUTS = 500

parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=DEFAULT_NUM_RUNS)
parser.add_argument('--agentname', type=str, default=DEFAULT_AGENT)
parser.add_argument('--envname', type=str, default=DEFAULT_ENV)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=DEFAULT_NUM_ROLLOUTS)
args = parser.parse_args()

start_time = get_time_string()
start_datetime = parse_time_string(start_time)

for i in range(args.num_runs):
    os.system("python {0} {1} {2} --log_dir_root={3} --num_rollouts={4}".format(RUN_FILE,
                                                                           args.agentname,
                                                                           args.envname,
                                                                           LOG_DIR_ROOT,
                                                                           args.num_rollouts))

print("Success")
