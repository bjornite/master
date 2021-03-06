# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import os
import datetime
import argparse
from utilities import get_time_string

#LOG_DIR_ROOT = "epsbootdqn_experiments"
LOG_DIR_ROOT = "/media/bjornivar/63B84F7A4C4AA554/Master/experiments"
cond = "agent"
agents = [
    "DDQN",
    "R",
    "KB",
    "CB",
    #"Thompson",
    #"BootDQN",
    #"EpsBootDQN",
    #"KBBoot",
    #"AllCombined",
]
learning_rates = [1e-3]
epsilon = [10000]
experiments = [#("CartPole-v0", 600, [8], "smallonelayernet"),
               #("CartPole-v0", 600, [8, 8], "smalltwolayernet"),
               #("CartPole-v0", 600, [8, 8, 8], "smallthreelayernet"),
               #("CartPole-v0", 600, [32], "largeonelayernet"),
               #("CartPole-v0", 600, [32, 32], "largetwolayernet"),
               #("CartPole-v0", 600, [32, 32, 32], "largethreelayernet"),
               ("MountainCar-v0", 1500, [32, 32], "largetwolayernet"),
               ("MountainCar-v0", 1500, [8], "smallonelayernet"),
               ("MountainCar-v0", 1500, [8, 8], "smalltwolayernet"),
               ("MountainCar-v0", 1500, [8, 8, 8], "smallthreelayernet"),
               ("MountainCar-v0", 1500, [32], "largeonelayernet"),
               ("MountainCar-v0", 1500, [32, 32, 32], "largethreelayernet"),
               ("MountainCarStochasticArea-v0", 1500, [32, 32], "largetwolayernet"),
               ("MountainCarStochasticArea-v0", 1500, [8], "smallonelayernet"),
               ("MountainCarStochasticArea-v0", 1500, [8, 8], "smalltwolayernet"),
               ("MountainCarStochasticArea-v0", 1500, [8, 8, 8], "smallthreelayernet"),
               ("MountainCarStochasticArea-v0", 1500, [32], "largeonelayernet"),
               ("MountainCarStochasticArea-v0", 1500, [32, 32, 32], "largethreelayernet"),
]

for i in range(len(experiments)):
    for lr in learning_rates:
        for eps in epsilon:
            env, rollouts, hiddens, ldir = experiments[i]
            log_dir = LOG_DIR_ROOT + "/" + ldir
            series_dict = {}
            counter = 0
            for subdir, dirs, files in os.walk(log_dir):
                for dir in sorted(dirs):
                    if dir.split("_")[1] == env:
                        for subdir2, dirs2, files2 in os.walk(os.path.join(log_dir, dir)):
                            for file in sorted(files2):
                                if file == "returns.csv":
                                    s = pd.read_csv(os.path.join(log_dir, dir, file),
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
                                    s["return"] = s["return"].rolling(100, min_periods=1).median()
                                    series_dict[counter] = s
                                    counter += 1
            try:
                df = pd.concat(series_dict, ignore_index=True)
            except:
                print("Could not concatenate dict to dataframe")
                continue
            noeps = ["Thompson",
                     "BootDQN",
                     "EpsBootDQN",
                     "KBBoot",
                     "AllCombined"]
            haseps = ["DDQN", "KB", "CB", "R"]
            df = df.loc[df['agent'].isin(agents) & df["epsilon"].isin(epsilon)]
            df.loc[df['agent'].isin(haseps), "agent"] = df["agent"].loc[df['agent'].isin(haseps)] + " k: " + df["epsilon"].loc[df['agent'].isin(haseps)].astype(str)
            df = df.loc[df['learning_rate'] == lr]
            #df['median'] = df['return'].rolling(100, center=False, min_periods=1).median()
            #df.plot()
            df = df.loc[df['iteration'] <= 1400]
            import latexipy as lp
            lp.latexify()  # Change to a serif font that fits with most LaTeX.
            txt = ldir + env + "_lr" + str(lr).replace('.',"_") + "_eps" + str(eps)
            savedir = "experiments/img/intrinsic/" + ldir + "/" + env
            with lp.figure(txt, directory=savedir, size = lp.figure_size(n_columns=1), exts=["pdf"]):  # saves in img/ by default.
                sns.tsplot(data=df,
                           time="iteration",
                           value="return",
                           condition=cond,
                           unit="run",
                           #ci=[5, 50, 90],
                           err_style=None,
                           estimator=np.nanmean)
                #estimator=np.nanmean)
                #plt.ylim([0, 210])
                plt.xlabel("Episode")
                plt.ylabel("Total reward")
                # Put the legend out of the figure
                #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
