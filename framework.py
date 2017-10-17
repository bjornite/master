import pickle
import go_vncdriver
import tensorflow as tf
import numpy as np
import random
import datetime
import time
import os
from shutil import copyfile
from basic_q_learning import Qlearner,  Random_agent, KBQlearner, IKBQlearner, CBQlearner
from utilities import get_time_string, get_log_dir, parse_time_string
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gym
#import universe
import git

def get_agent(name, env, log_dir, learning_rate, reg_beta):
    if name == "Qlearner":
        return Qlearner(name, env, log_dir, learning_rate, reg_beta)
    elif name == "KBQlearner":
        return KBQlearner(name, env, log_dir, learning_rate, reg_beta)
    elif name == "IKBQlearner":
        return IKBQlearner(name, env, log_dir, learning_rate, reg_beta)
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

    repo = git.Repo(search_parent_directories=True)
    label = repo.head.object.hexsha + "\n" + repo.head.object.message
    print label
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
    with open('{}/code_version.txt'.format(log_dir), 'w') as f:
        f.write(label)

    returns = []
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    max_score = 497
    num_test_runs = 15

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
    lr_update_step = agent.model.learning_rate * ((1.0/(args.num_rollouts*0.9)))
    rp_update_step = agent.random_action_prob * ((1.0/(args.num_rollouts*0.9)))
    stop_training = False
    for i in range(args.num_rollouts):
        state = env.reset()
        action = env.action_space.sample()
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
        if agent.model.learning_rate > 1e-5:
            agent.model.learning_rate -= lr_update_step
        if agent.random_action_prob > 0:
            agent.random_action_prob -= rp_update_step
        while not done:
            double_state = np.concatenate([last_state, state])
            action = agent.get_action(double_state)
            log_action = action
            if args.random_cartpole and (state[0] > 0.2):
                action = env.action_space.sample()
            obs, r, done, _ = env.step(action)
            if done:
                r = -1
                obs = np.zeros(env.observation_space.shape[0])
            agent.replay_memory.append((double_state,
                                        log_action,
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
            current_weights = agent.model.get_weights()
            count = 0
            for w in agent.old_weights:
                agent.old_weights[count] = np.add(np.multiply(agent.old_weights[count], 0.999),
                                              np.multiply(current_weights[count], (1-0.999)))
                if agent.old_weights[count].shape[0] == 1:
                    agent.old_weights[count] = agent.old_weights[count].reshape([-1])
                count += 1
            if len(agent.replay_memory) > agent.minibatch_size and not stop_training:
                agent.train(args.no_tf_log)
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
                    if args.render:
                        env.render()
            test_results.append(totalr / num_test_runs)
            print("iter {0}, reward: {1:.2f}, lr: {2}, rp: {3}".format(i,
                                                                       totalr/num_test_runs,
                                                                       agent.model.learning_rate,
                                                                       agent.random_action_prob))
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
