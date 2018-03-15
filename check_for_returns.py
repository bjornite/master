import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import os
import datetime
import argparse
from utilities import get_time_string

LOG_DIR_ROOT = "/media/bjornivar/63B84F7A4C4AA554/Master/experiments"
cond = "agent"
agents = [
    "DDQN",
    "R",
    "KB",
    "CB",
    "Thompson",
    "BootDQN",
    "KBBoot"]
learning_rates = [1e-3, 5e-3]
epsilon = [10000, 1000, 100]
experiments = [("CartPole-v0", 600, [8, 8], "smallonelayernet"),
               ("CartPole-v0", 600, [8, 8], "smalltwolayernet"),
               ("CartPole-v0", 600, [8, 8, 8], "smallthreelayernet"),
               ("CartPole-v0", 600, [32], "largeonelayernet"),
               ("CartPole-v0", 600, [32, 32], "largetwolayernet"),
               ("CartPole-v0", 600, [32, 32, 32], "largethreelayernet"),
               ("MountainCar-v0", 1500, [8], "smallonelayernet"),
               ("MountainCar-v0", 1500, [8, 8], "smalltwolayernet"),
               ("MountainCar-v0", 1500, [8, 8, 8], "smallthreelayernet"),
               ("MountainCar-v0", 1500, [32], "largeonelayernet"),
               ("MountainCar-v0", 1500, [32, 32], "largetwolayernet"),
               ("MountainCar-v0", 1500, [32, 32, 32], "largethreelayernet")]

for i in range(len(experiments)):
    for agent in agents:
        env, rollouts, hiddens, ldir = experiments[i]
        log_dir = LOG_DIR_ROOT + "/" + ldir
        series_dict = {}
        counter = 0
        for subdir, dirs, files in os.walk(log_dir):
            for dir in sorted(dirs):
                if dir.split("_")[1] == env and dir.split("_")[0] == agent:
                    for subdir2, dirs2, files2 in os.walk(os.path.join(log_dir, dir)):
                        for file in sorted(files2):
                            if file == "returns.csv":
                                counter += 1
        print(ldir + "\t" + env + "\t" + agent  + "\t\t" + str(counter))
