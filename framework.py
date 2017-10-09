import pickle
import go_vncdriver
import tensorflow as tf
import numpy as np
import random
import datetime
import time
import os
from shutil import copyfile
from basic_q_learning import Qlearner,  Random_agent, KBQlearner, CBQlearner
from utilities import get_time_string, get_log_dir, parse_time_string
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gym
import universe


def get_agent(name, env, log_dir, learning_rate, reg_beta):
    if name == "Qlearner":
        return Qlearner(name, env, log_dir, learning_rate, reg_beta)
    elif name == "KBQlearner":
        return KBQlearner(name, env, log_dir, learning_rate, reg_beta)
    elif name == "CBQlearner":
        return CBQlearner(name, env, log_dir, learning_rate, reg_beta)
    elif name == "Random_agent":
        return Random_agent(name, env, log_dir)
    else:
        print("No agent type named {0}".format(name))


if __name__ == "__main__":
    # Load the kind of agent currently being tested
    # Run through testing regime, specify simulations and number of runs per
    # simulation

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('agentname', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--log_dir_root', type=str, default="logfiles")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--random_cartpole', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.)
    parser.add_argument('--regularization_beta', type=float, default=0.)
    parser.add_argument('--no_tf_log', action='store_true')
    args = parser.parse_args()
    log_dir = get_log_dir(args.agentname, args.envname, args.log_dir_root)
    try:
        os.mkdir(log_dir)
    except:
        pass
    copyfile("tf_neural_net.py", "{}/tf_neural_net.py".format(log_dir))
    copyfile("basic_q_learning.py", "{}/basic_q_learning.py".format(log_dir))

    returns = []
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    print('Initializing agent')
    try:
        tf.get_default_session().close()
    except AttributeError:
        pass
    agent = get_agent(args.agentname, env, log_dir, args.learning_rate, args.regularization_beta)
    learning_rate = agent.model.learning_rate
    if args.random_cartpole:
        args.envname = "CartPole-v1-random"
    print('Initialized')
    rewards = []
    observations = []
    actions = []
    global_steps = 0
    for i in range(args.num_rollouts):
        state = env.reset()
        local_observations = []
        local_actions = []
        local_rewards = []
        done = False
        totalr = 0.
        steps = 0
        mean_cb_r = 0
        while not done:
            action = agent.get_action(state)
            if args.random_cartpole and (state[0] > 0.2):
                action = env.action_space.sample()
            actions.append(action)
            local_actions.append(action)
            obs, r, done, _ = env.step(action)
            observations.append(obs)
            local_observations.append(obs)
            local_rewards.append(r)
            rewards.append(r)
            agent.replay_memory.append((state, action, obs, r, done))
            state = obs
            if len(agent.replay_memory) > agent.replay_memory_size:
                agent.replay_memory.pop(0)
            totalr += r
            steps += 1
            global_steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
            if global_steps % agent.target_update_freq == 0:
                current_weights = agent.model.get_weights()
                agent.old_weights = current_weights
            if len(agent.replay_memory) > agent.minibatch_size:
                mean_cb_r = agent.train(args.no_tf_log)
        returns.append(totalr)
        print("iter {0}, reward: {1:.2f}, lr: {2}".format(i,
                                                          totalr,
                                                          learning_rate))
    agent.model.sess.close()
    log_data = pd.DataFrame()
    log_data["return"] = returns
    log_data["agent"] = [args.agentname]*len(log_data)
    log_data["env"] = [args.envname]*len(log_data)
    log_data["learning_rate"] = learning_rate*len(log_data)
    log_data.to_csv("{0}/returns.csv".format(log_dir))
