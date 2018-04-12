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
agents = ["DDQN",
          #"R",
          "KB",
          #"CB",
          "Thompson",
          "BootDQN",
          #"EpsBootDQN",
          #"KBBoot",
          #"AllCombined",
]
learning_rates = [1e-3]
epsilon = [10000, 100]
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
               #("MountainCar-v0", 1500, [32, 32, 32], "largethreelayernet"),
               #("MountainCarStochasticArea-v0", 1500, [8], "smallonelayernet"),
               #("MountainCarStochasticArea-v0", 1500, [8, 8], "smalltwolayernet"),
               #("MountainCarStochasticArea-v0", 1500, [8, 8, 8], "smallthreelayernet"),
               #("MountainCarStochasticArea-v0", 1500, [32], "largeonelayernet"),
               #("MountainCarStochasticArea-v0", 1500, [32, 32], "largetwolayernet"),
               #("MountainCarStochasticArea-v0", 1500, [32, 32, 32], "largethreelayernet"),
]

res = {}

for i in range(len(experiments)):
    counter = 0
    series_dict = {}
    env, rollouts, hiddens, ldir = experiments[i]
    res[str(hiddens)] = res.get(str(hiddens), {})
    log_dir = LOG_DIR_ROOT + "/" + ldir
    for subdir, dirs, files in os.walk(log_dir):
        for dir in sorted(dirs):
            if dir.split("_")[1] != env:
                continue
            log_dir2 = os.path.join(log_dir, dir)
            for subdir2, dirs2, files2 in os.walk(log_dir2):
                for file in sorted(files2):
                    if file == "returns.csv":
                        s = pd.read_csv(os.path.join(log_dir2, file),
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
    #print(env + ":")
    for lr in learning_rates:
        #print("\tlr={}:".format(lr))
        res[str(hiddens)][lr] = {}
        #print("\t\teps={}:".format(eps))
        for agent in agents:
            res[str(hiddens)][lr][agent] = {}
            noeps = ["Thompson",
                     "BootDQN",
                     "EpsBootDQN",
                     "KBBoot",
                     "AllCombined"]
            haseps = ["DDQN", "KB", "CB", "R"]
            try:
                df = pd.concat(series_dict, ignore_index=True)
            except:
                continue
            df = df.loc[df['learning_rate'] == lr]
            df = df.loc[df["agent"] == agent]
            df = df.loc[df["epsilon"].isin(epsilon)]
            df.loc[df['agent'].isin(haseps), "agent"] = df["agent"].loc[df['agent'].isin(haseps)] + " k: " + df["epsilon"].loc[df['agent'].isin(haseps)].astype(str)
            res[str(hiddens)][lr][agent] = {}

            if env == "CartPole-v0":
                cutoff = 300
            elif env == "MountainCar-v0":
                cutoff = 1500
            elif env == "MountainCarStochasticArea-v0":
                cutoff = 1500
            else:
                print("Must add cutoff value for this environment: {}".format(env))
                sys.exit(0)
            res[str(hiddens)][lr][agent]['returns'] = []
            res[str(hiddens)][lr][agent]['highscores'] = []
            res[str(hiddens)][lr][agent]['mean_best_streak'] = []
            for run in set(df['run']):
                df_run = df.loc[df['run'] == run]
                # Calculate sum of returns
                returns = df_run.loc[df_run['iteration'] <= cutoff]['return'].sum()
                # Calculate average of 90th percentile episodes
                highscores = df_run.loc[df_run['return'] >= df_run['return'].quantile(.90)]['return'].mean()
                maxscore = df_run['return'].max()
                res[str(hiddens)][lr][agent]['returns'].append(returns)
                res[str(hiddens)][lr][agent]['highscores'].append(highscores)
                res[str(hiddens)][lr][agent]['mean_best_streak'].append(df_run['return'].rolling(100).mean().max())
                #res[env][str(hiddens)][lr][eps][agent]['maxscore'] = maxscore
                #print("\t\t\t{0}:\t{1:.0f}\t{2:.2f}\t{3}".format(agent, returns, highscores, maxscore))
# Normalize scores relative to DDQN for each parameter setting
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


for arch, data2 in res.items():
    best_ret = float('-inf')
    best_high = float('-inf')
    for lr, data3 in data2.items():
        ret = np.nanmean(data3["DDQN"]["returns"])
        if ret > best_ret: best_ret = ret
        high = np.nanmean(data3["DDQN"]["highscores"])
        if high > best_high: best_high = high
for arch, data2 in res.items():
    for lr, data3 in data2.items():
        for agent, data4 in data3.items():
            #maxs = data4["DDQN"]["maxscore"]
            data4['returns'] = '{0:.2f} ±{1:.2f}'.format(np.nanmean(data4["returns"]),# / abs(best_ret),
                                                         np.nanstd(data4["returns"])) # / abs(best_ret))
            data4['highscores'] = '{0:.2f} ±{1:.2f}'.format(np.nanmean(data4["highscores"]), # / abs(best_high),
                                                            np.nanstd(data4["highscores"])) # / abs(best_high))
            data4['mean_best_streak'] = '{0:.2f} ±{1:.2f}'.format(np.nanmean(data4["mean_best_streak"]), # / abs(best_high),
                                                                  np.nanstd(data4["mean_best_streak"])) # / abs(best_high))

# Write results to file:
reform = {(arch, lr, agent): data3 for arch, data in res.items() for lr, data2 in data.items() for agent, data3 in data2.items()}
df = pd.DataFrame.from_dict(reform, orient="index")
mi = pd.MultiIndex.from_tuples(df.index, names=["layers", "lr", "agent"])
df.index = mi
def formater(x):
    return "{0:.2f}".format(x)
print(df.to_latex(float_format=formater))
#with open("calculated_results.txt", "w+") as f:
#    for env, data in res.items():
#        print("\\begin{table}\n\\centering\n\\begin{tabular}{ |c|@{}c@{}| }\n\\hline\nno test & \\begin{tabular}{|c|@{}c@{}| } no test \\\\ \hline 1.23 \n \\end{tabular} \\\\")
#        for arch, data2 in data.items():
#            for lr, data3 in data2.items():
#                for eps, data4 in data3.items():
#                    for agent, data5 in data4.items():
#                        data5["returns"] /= ret
#                        data5["highscores"] /= high
#                        data5["maxscore"] /= maxs
#    print("\\hline\n\\end{tabular}\n\\end{table}")
