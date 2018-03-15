import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import sys
import os
import datetime
import argparse
from utilities import get_time_string

LOG_DIR_ROOT = "experiments"
#LOG_DIR_ROOT = "/media/bjornivar/63B84F7A4C4AA554/Master/experiments"
cond = "agent"
agents = [
    "DDQN",
    #"R",
    "KB",
    #"CB",
    "Thompson",
    "BootDQN",
    "KBBoot"]
learning_rates = [1e-3, 5e-3]
epsilon = [10000, 1000]
experiments = [("CartPole-v0", 600, [8], "smallonelayernet"),
               #("CartPole-v0", 600, [8, 8], "smalltwolayernet"),
               #("CartPole-v0", 600, [8, 8, 8], "smallthreelayernet"),
               #("CartPole-v0", 600, [32], "largeonelayernet"),
               #("CartPole-v0", 600, [32, 32], "largetwolayernet"),
               #("CartPole-v0", 600, [32, 32, 32], "largethreelayernet"),
               #("MountainCar-v0", 1500, [8], "smallonelayernet"),
               #("MountainCar-v0", 1500, [8, 8], "smalltwolayernet"),
               #("MountainCar-v0", 1500, [8, 8, 8], "smallthreelayernet"),
               #("MountainCar-v0", 1500, [32], "largeonelayernet"),
               #("MountainCar-v0", 1500, [32, 32], "largetwolayernet"),
               #("MountainCar-v0", 1500, [32, 32, 32], "largethreelayernet")
]

log_dir = LOG_DIR_ROOT
series_dict = {}
counter = 0
for subdir, dirs, files in os.walk(log_dir):
    for dir in sorted(dirs):
        log_dir2 = os.path.join(log_dir, dir)
        for subdir2, dirs2, files2 in os.walk(log_dir2):
            for dir2 in sorted(dirs2):
                log_dir3 = os.path.join(log_dir2, dir2)
                for subdir3, dirs3, files3 in os.walk(log_dir3):
                    for file in sorted(files3):
                        if file == "returns.csv":
                            s = pd.read_csv(os.path.join(log_dir2, dir2, file),
                                            header=None,
                                            names=["iteration",
                                                   "return",
                                                   "agent",
                                                   "env",
                                                   "learning_rate",
                                                   "regularization_beta",
                                                   "epsilon"],
                                            skiprows=1)
                            s["run"] = [counter] * len(s)
                            series_dict[counter] = s
                            counter += 1

for i in range(len(experiments)):
    env, rollouts, hiddens, ldir = experiments[i]
    print(env)
    for lr in learning_rates:
        print("\tlr={}:".format(lr))
        for eps in epsilon:
            print("\t\teps={}:".format(eps))
            for agent in agents:
                df = pd.concat(series_dict, ignore_index=True)
                df = df.loc[df['epsilon'] == eps]
                df = df.loc[df['learning_rate'] == lr]
                df = df.loc[df['agent'] == agent]
                #df.plot()
                #df = df.loc[df['iteration'] <= 600]
                # Calculate sum of returns
                if env == "CartPole-v0":
                    cutoff = 300
                elif env == "MountainCar-v0":
                    cutoff = 1500
                else:
                    print("Must add cutoff value for this environment: {}".format(env))
                    sys.exit(0)
                returns = df.loc[df['iteration'] <= cutoff]['return'].sum()
                # Calculate average of 90th percentile episodes
                highscores = df.loc[df['return'] > df['return'].quantile(.90)]['return'].mean()
                print("\t\t\t{0}:\t{1:.0f}\t{2:.2f}".format(agent, returns, highscores))
