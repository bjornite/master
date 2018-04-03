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

#LOG_DIR_ROOT = "experiments"
LOG_DIR_ROOT = "/media/bjornivar/63B84F7A4C4AA554/Master/experiments"
cond = "agent"
agents = ["DDQN",
          #"R",
          "KB",
          #"CB",
          "Thompson",
          "BootDQN",
          "KBBoot",
          #"AllCombined",
]
learning_rates = [1e-3, 5e-3]
epsilon = [10000, 1000]
experiments = [#("CartPole-v0", 600, [8], "smallonelayernet"),
               #("CartPole-v0", 600, [8, 8], "smalltwolayernet"),
               #("CartPole-v0", 600, [8, 8, 8], "smallthreelayernet"),
               #("CartPole-v0", 600, [32], "largeonelayernet"),
               #("CartPole-v0", 600, [32, 32], "largetwolayernet"),
               #("CartPole-v0", 600, [32, 32, 32], "largethreelayernet"),
               ("MountainCar-v0", 1500, [8], "smallonelayernet"),
               ("MountainCar-v0", 1500, [8, 8], "smalltwolayernet"),
               ("MountainCar-v0", 1500, [8, 8, 8], "smallthreelayernet"),
               ("MountainCar-v0", 1500, [32], "largeonelayernet"),
               ("MountainCar-v0", 1500, [32, 32], "largetwolayernet"),
               ("MountainCar-v0", 1500, [32, 32, 32], "largethreelayernet")
]

res = {}

for i in range(len(experiments)):
    counter = 0
    series_dict = {}
    env, rollouts, hiddens, ldir = experiments[i]
    res[env] = res.get(env, {})
    res[env][str(hiddens)] = {}
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
        res[env][str(hiddens)][lr] = {}
        #print("\t\teps={}:".format(eps))
        for agent in agents:
            res[env][str(hiddens)][lr][agent] = {}
            if agent in ["DDQN", "R", "KB", "CB"]:
                for eps in epsilon:
                    res[env][str(hiddens)][lr][agent][eps] = {}
                    df = pd.concat(series_dict, ignore_index=True)
                    df = df.loc[df['epsilon'] == eps]
                    df = df.loc[df['learning_rate'] == lr]
                    df = df.loc[df['agent'] == agent]
                    #print(agent)
                    #df.plot()
                    #df = df.loc[df['iteration'] <= 600]
                    if env == "CartPole-v0":
                        cutoff = 300
                    elif env == "MountainCar-v0":
                        cutoff = 1500
                    else:
                        print("Must add cutoff value for this environment: {}".format(env))
                        sys.exit(0)
                    res[env][str(hiddens)][lr][agent][eps]['returns'] = []
                    res[env][str(hiddens)][lr][agent][eps]['highscores'] = []
                    res[env][str(hiddens)][lr][agent][eps]['mean_best_streak'] = []
                    for run in set(df['run']):
                        df_run = df.loc[df['run'] == run]
                        # Calculate sum of returns
                        returns = df_run.loc[df_run['iteration'] <= cutoff]['return'].sum()
                        # Calculate average of 90th percentile episodes
                        highscores = df_run.loc[df_run['return'] >= df_run['return'].quantile(.90)]['return'].mean()
                        maxscore = df_run['return'].max()
                        res[env][str(hiddens)][lr][agent][eps]['returns'].append(returns)
                        res[env][str(hiddens)][lr][agent][eps]['highscores'].append(highscores)
                        res[env][str(hiddens)][lr][agent][eps]['mean_best_streak'].append(df_run['return'].rolling(100).mean().max())
                        #res[env][str(hiddens)][lr][eps][agent]['maxscore'] = maxscore
                        #print("\t\t\t{0}:\t{1:.0f}\t{2:.2f}\t{3}".format(agent, returns, highscores, maxscore))
            else:
                eps = "N/A"
                res[env][str(hiddens)][lr][agent][eps] = {}
                df = pd.concat(series_dict, ignore_index=True)
                df = df.loc[df['learning_rate'] == lr]
                df = df.loc[df['agent'] == agent]
                #print(agent)
                #df.plot()
                #df = df.loc[df['iteration'] <= 600]
                if env == "CartPole-v0":
                    cutoff = 300
                elif env == "MountainCar-v0":
                    cutoff = 1500
                else:
                    print("Must add cutoff value for this environment: {}".format(env))
                    sys.exit(0)
                res[env][str(hiddens)][lr][agent][eps]['returns'] = []
                res[env][str(hiddens)][lr][agent][eps]['highscores'] = []
                res[env][str(hiddens)][lr][agent][eps]['mean_best_streak'] = []
                for run in set(df['run']):
                    df_run = df.loc[df['run'] == run]
                    # Calculate sum of returns
                    returns = df_run.loc[df_run['iteration'] <= cutoff]['return'].sum()
                    # Calculate average of 90th percentile episodes
                    highscores = df_run.loc[df_run['return'] >= df_run['return'].quantile(.90)]['return'].mean()
                    maxscore = df_run['return'].max()
                    res[env][str(hiddens)][lr][agent][eps]['returns'].append(returns)
                    res[env][str(hiddens)][lr][agent][eps]['highscores'].append(highscores)
                    res[env][str(hiddens)][lr][agent][eps]['mean_best_streak'].append(df_run['return'].rolling(100).mean().max())
                    #res[env][str(hiddens)][lr][eps][agent]['maxscore'] = maxscore
                    #print("\t\t\t{0}:\t{1:.0f}\t{2:.2f}\t{3}".format(agent, returns, highscores, maxscore))
# Normalize scores relative to DDQN for each parameter setting
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

for env, data in res.items():
    best_ret = float('-inf')
    best_high = float('-inf')
    for arch, data2 in data.items():
        for lr, data3 in data2.items():
            for eps, data_ddqn in data3["DDQN"].items():
                ret = np.nanmean(data3["DDQN"][eps]["returns"])
                if ret > best_ret: best_ret = ret
                high = np.nanmean(data3["DDQN"][eps]["highscores"])
                if high > best_high: best_high = high
    for arch, data2 in data.items():
        for lr, data3 in data2.items():
            for agent, data4 in data3.items():
                for eps, data5 in data4.items():
                #maxs = data4["DDQN"]["maxscore"]
                    data5['returns'] = '{0:.2f} ±{1:.2f}'.format(np.nanmean(data5["returns"]),# / abs(best_ret),
                                                          np.nanstd(data5["returns"])) # / abs(best_ret))
                    data5['highscores'] = '{0:.2f} ±{1:.2f}'.format(np.nanmean(data5["highscores"]), # / abs(best_high),
                                                          np.nanstd(data5["highscores"])) # / abs(best_high))
                    data5['mean_best_streak'] = '{0:.2f} ±{1:.2f}'.format(np.nanmean(data5["mean_best_streak"]), # / abs(best_high),
                                                          np.nanstd(data5["mean_best_streak"])) # / abs(best_high))

# Write results to file:
reform = {(env, arch, lr, agent, eps): data5 for env, data in res.items() for arch, data2 in data.items() for lr, data3 in data2.items() for agent, data4 in data3.items() for eps, data5 in data4.items()}
df = pd.DataFrame.from_dict(reform, orient="index")
mi = pd.MultiIndex.from_tuples(df.index, names=["env", "layers", "lr", "agent", "epsilon-schedule"])
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
