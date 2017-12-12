import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import os
import datetime
import argparse
from utilities import get_time_string

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str)
    parser.add_argument('--condition', type=str, default="agent")
    parser.add_argument('--envname', type=str, default="CartPole-v1")
    parser.add_argument('--agentname', type=str, nargs="*", default=[])
    parser.add_argument('--datetime_low', type=str, default="")
    parser.add_argument('--datetime_high', type=str, default=get_time_string())
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    log_dir = args.log_dir

    series_dict = {}
    counter = 0

    for subdir, dirs, files in os.walk(log_dir):
        for dir in dirs:
            if dir.split("_")[1] == args.envname:
                for subdir2, dirs2, files2 in os.walk(os.path.join(log_dir, dir)):
                    for file in files2:
                        if file == "returns.csv":
                            s = pd.read_csv(os.path.join(log_dir, dir, file),
                                            header=None,
                                            names=["iteration", "return", "agent", "env", "learning_rate", "regularization_beta", "test_results"],
                                            skiprows=1)
                            s["run"] = [counter] * len(s)
                            series_dict[counter] = s
                            counter += 1

    df = pd.concat(series_dict, ignore_index=True)
    if args.agentname != []:
        df = df.loc[df['agent'].isin(args.agentname)]
    #df.plot()
    import latexipy as lp

    lp.latexify()  # Change to a serif font that fits with most LaTeX.
    if log_dir[-1] is not '/':
        txt = log_dir.split('/')
    else:
        txt = log_dir[:-1].split('/')
    with lp.figure(txt[-1]):  # saves in img/ by default.
        sns.tsplot(data=df,
                   time="iteration",
                   value="return",
                   condition=args.condition,
                   unit="run",
                   ci="sd",
                   #err_style="unit_traces",
                   estimator=np.nanmean)
        plt.xlabel("Episode")
