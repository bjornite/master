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
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--regularization_beta', type=float, default=0.)
    parser.add_argument('--no_tf_log', action='store_true', default=False)
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

    num_test_runs=15

    print('Initializing agent')
    try:
        tf.get_default_session().close()
    except AttributeError:
        pass
    agent = get_agent(args.agentname, env, log_dir, args.learning_rate, args.regularization_beta)
    learning_rate = agent.model.learning_rate
    reg_beta = agent.model.beta
    if args.random_cartpole:
        args.envname = "CartPole-v1-random"
    print('Initialized')
    rewards = []
    observations = []
    actions = []
    test_results = []
    global_steps = 0
    for i in range(args.num_rollouts):
        state = env.reset()
        action = env.action_space.sample()
        actions.append(action)
        obs, r, done, _ = env.step(action)
        last_state = state
        state = obs
        local_observations = []
        local_actions = []
        local_rewards = []
        done = False
        totalr = 0.
        steps = 0
        mean_cb_r = 0
        while not done:
            double_state = np.concatenate([last_state, state])
            action = agent.get_action(double_state)
            if args.random_cartpole and (state[0] > 0.2):
                action = env.action_space.sample()
            obs, r, done, _ = env.step(action)
            if done:
                r = -1
            agent.replay_memory.append((double_state,
                                        action,
                                        np.concatenate([state, obs]),
                                        r,
                                        done))
            last_state = state
            state = obs
            if len(agent.replay_memory) > agent.replay_memory_size:
                agent.replay_memory.pop(0)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
            if agent.model.global_step % agent.target_update_freq == 0:
                current_weights = agent.model.get_weights()
                agent.old_weights = current_weights
            if len(agent.replay_memory) > agent.minibatch_size:
                mean_cb_r = agent.train(args.no_tf_log)
        returns.append(totalr)
        if i % (args.num_rollouts / 100) == 0:
            totalr = 0.
            for j in range(num_test_runs):
                state = env.reset()
                action = env.action_space.sample()
                obs, r, done, _ = env.step(action)
                last_state = state
                state = obs
                while not done:
                    double_state = np.concatenate([last_state, state])
                    action = agent.get_action(double_state, is_test=True)
                    if args.random_cartpole and (state[0] > 0.2):
                        action = env.action_space.sample()
                    obs, r, done, _ = env.step(action)
                    if done:
                        r = -1
                    last_state = state
                    state = obs
                    totalr += r
                    env.render()
            test_results.append(totalr / num_test_runs)
            print("iter {0}, reward: {1:.2f}, lr: {2}, beta: {3}".format(i,
                                                                         totalr/num_test_runs,
                                                                         learning_rate,
                                                                         reg_beta))
        else:
            test_results.append(None)
    agent.model.sess.close()
    log_data = pd.DataFrame()
    log_data["return"] = returns
    log_data["agent"] = [args.agentname]*len(log_data)
    log_data["env"] = [args.envname]*len(log_data)
    log_data["learning_rate"] = [learning_rate]*len(log_data)
    log_data["regularization_beta"] = [reg_beta]*len(log_data)
    log_data["test_results"] = test_results
    log_data.to_csv("{0}/returns.csv".format(log_dir))
