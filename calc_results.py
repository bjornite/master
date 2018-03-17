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
          "R",
          "KB",
          "CB",
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

res = {}

for i in range(len(experiments)):
    counter = 0
    series_dict = {}
    env, rollouts, hiddens, ldir = experiments[i]
    res[env] = {}
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
        for eps in epsilon:
            res[env][str(hiddens)][lr][eps] = {}
            #print("\t\teps={}:".format(eps))
            for agent in agents:
                res[env][str(hiddens)][lr][eps][agent] = {}
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
                res[env][str(hiddens)][lr][eps][agent]['returns'] = []
                res[env][str(hiddens)][lr][eps][agent]['highscores'] = []
                for run in set(s['run']):
                    # Calculate sum of returns
                    returns = df.loc[df['iteration'] <= cutoff]['return'].sum()
                    # Calculate average of 90th percentile episodes
                    highscores = df.loc[df['return'] >= df['return'].quantile(.90)]['return'].mean()
                    maxscore = df['return'].max()
                    res[env][str(hiddens)][lr][eps][agent]['returns'].append(returns)
                    res[env][str(hiddens)][lr][eps][agent]['highscores'].append(highscores)
                    #res[env][str(hiddens)][lr][eps][agent]['maxscore'] = maxscore
                    #print("\t\t\t{0}:\t{1:.0f}\t{2:.2f}\t{3}".format(agent, returns, highscores, maxscore))

# Normalize scores relative to DDQN for each parameter setting
for env, data in res.items():
    for arch, data2 in data.items():
        for lr, data3 in data2.items():
            for eps, data4 in data3.items():
                ret = np.average(data4["DDQN"]["returns"])
                high = np.average(data4["DDQN"]["highscores"])
                #maxs = data4["DDQN"]["maxscore"]
                for agent, data5 in data4.items():
                    data5["returns"] = np.average(data5["returns"]) / ret
                    data5["highscores"] = np.average(data5["highscores"]) / high
                    #data5["maxscore"] /= maxs

# Write results to file:
reform = {(env, arch, lr, eps, agent): data5 for env, data in res.items() for arch, data2 in data.items() for lr, data3 in data2.items() for eps, data4 in data3.items() for agent, data5 in data4.items()}
df = pd.DataFrame.from_dict(reform, orient="index")
mi = pd.MultiIndex.from_tuples(df.index, names=["env", "layers", "lr", "$\\epsilon$-schedule", "agent"])
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
