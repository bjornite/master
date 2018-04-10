import pickle
import go_vncdriver
import tensorflow as tf
import numpy as np
import random
import datetime
import time
import os
import sys
import json
from shutil import copyfile
from basic_q_learning import DDQN,  Random_agent, KB, IKBQlearner, CB, SAQlearner, ISAQlearner, MSAQlearner, IMSAQlearner, TESTQlearner, R
from modular_q_learning import BootDQN, EpsBootDQN, KBBoot, CBBoot, Thompson, AllCombined
from utilities import get_time_string, get_log_dir, parse_time_string
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gym
import deepexplorebenchmark
#import universe
import git

def get_agent(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon):
    if name == "DDQN":
        return DDQN(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "KB":
        return KB(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "IKBQlearner":
        return IKBQlearner(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "CB":
        return CB(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "R":
        return R(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "SAQlearner":
        return SAQlearner(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "ISAQlearner":
        return ISAQlearner(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "MSAQlearner":
        return MSAQlearner(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "IMSAQlearner":
        return IMSAQlearner(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "TESTQlearner":
        return TESTQlearner(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "Random_agent":
        return Random_agent(name, env, log_dir, n_hiddens, epsilon)
    elif name == "BootDQN":
        return BootDQN(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "EpsBootDQN":
        return EpsBootDQN(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "KBBoot":
        return KBBoot(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "CBBoot":
        return CBBoot(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "Thompson":
        return Thompson(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    elif name == "AllCombined":
        return AllCombined(name, env, log_dir, learning_rate, reg_beta, n_hiddens, epsilon)
    else:
        print("No agent type named {0}".format(name))


if __name__ == "__main__":
    # Load the kind of agent currently being tested
    # Run through testing regime, specify simulations and number of runs per
    # simulation

    repo = git.Repo(search_parent_directories=True)
    label = repo.head.object.hexsha + "\n" + repo.head.object.message
    print(label)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('agentname', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--log_dir_root', type=str, default="logfiles")
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--regularization_beta', type=float, default=0.)
    parser.add_argument('--n_hiddens', nargs="+", type=int, default=[8])
    parser.add_argument('--epsilon', type=int, default=1000)
    parser.add_argument('--no_tf_log', action='store_true', default=False)
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--atari', action='store_true', default=False)
    args = parser.parse_args()
    log_dir = get_log_dir(args.agentname, args.envname, args.log_dir_root)
    try:
        os.mkdir(log_dir)
    except:
        pass
    with open('{}/code_version.txt'.format(log_dir), 'w') as f:
        f.write(label)

    returns = []
    if args.atari:
        print("System set for Atari-ram")
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    print('Initializing agent')
    agent = get_agent(args.agentname, env, log_dir, args.learning_rate, args.regularization_beta, args.n_hiddens, epsilon=args.epsilon)
    stop_training = False
    if args.model is not "":
        agent.load_model(args.model)
        stop_training = True
    print('Initialized')
    sarslist = []
    test_results = []
    global_steps = 0
    for i in range(args.num_rollouts):
        state = env.reset()
        if args.atari:
            state = (state/255.0) - 0.5
        done = False
        totalr = 0.
        steps = 0
        mean_cb_r = 0
        while not done:
            action = agent.get_action(state)
            obs, r, done, _ = env.step(action)
            totalr += r
            if args.atari:
                obs = (obs/255.0) - 0.5
                r = max(-1, min(1, r))
            sars = (state, action, obs, r, done)
            agent.remember(state, action, r, obs, done)
            sarslist.append(sars)
            state = obs
            steps += 1
            if args.render:
                img = env.render(mode="rgb_array")
                if i == 3:
                    import scipy
                    scipy.misc.imsave('mountaincar.pdf', img)
                    #img[:, 325, :] = 0
                    for i in range(len(img[:, 0, 0])):
                        for j in range(len(img[0, :, 0])):
                            for k in range(len(img[0, 0, :])):
                                if j > 600 *((-0.6+1.2)/1.8)  and j < 600 *((-0.3+1.2)/1.8) and img[i, j, k] == 255:
                                    img[i, j, k] = 200
                    scipy.misc.imsave('mountaincarstochastic.pdf', img)
                    print("Saved image")
            if steps >= max_steps:
                break
            if not stop_training:
                agent.train(args.no_tf_log)
                global_steps += 1
        returns.append(totalr)
        #if i % (args.num_rollouts / 100) == 0:
            #agent.plot_state_visits()
            #agent.save_model(log_dir, "{}_percent.ckpt".format(i / (args.num_rollouts / 100)))
        print("iter {0}, reward: {1:.2f} {2} {3}".format(i, totalr, agent.debug_string(), args.agentname))
        test_results.append(None)
    learning_rate = 0
    reg_beta = 0
    log_data = pd.DataFrame()
    log_data["return"] = returns
    log_data["agent"] = [args.agentname]*len(log_data)
    log_data["env"] = [args.envname]*len(log_data)
    log_data["learning_rate"] = [args.learning_rate]*len(log_data)
    log_data["regularization_beta"] = [reg_beta]*len(log_data)
    log_data["epsilon"] = [args.epsilon]*len(log_data)

    log_data.to_csv("{0}/returns.csv".format(log_dir))
    with open("{0}/trajectories.pkl".format(log_dir), 'wb+') as f:
        pickle.dump(sarslist, f)
    sys.exit(0)
