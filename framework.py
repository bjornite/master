import pickle
import go_vncdriver
import tensorflow as tf
import numpy as np
import random
import datetime
import time
import os
from shutil import copyfile
from basic_q_learning import Qlearner,  Random_agent, KBQlearner, IKBQlearner, CBQlearner, SAQlearner, ISAQlearner, MSAQlearner, IMSAQlearner, TESTQlearner
from utilities import get_time_string, get_log_dir, parse_time_string
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gym
import deepexplorebenchmark
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
    elif name == "SAQlearner":
        return SAQlearner(name, env, log_dir, learning_rate, reg_beta)
    elif name == "ISAQlearner":
        return ISAQlearner(name, env, log_dir, learning_rate, reg_beta)
    elif name == "MSAQlearner":
        return MSAQlearner(name, env, log_dir, learning_rate, reg_beta)
    elif name == "IMSAQlearner":
        return IMSAQlearner(name, env, log_dir, learning_rate, reg_beta)
    elif name == "TESTQlearner":
        return TESTQlearner(name, env, log_dir, learning_rate, reg_beta)
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
    print(label)
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
    parser.add_argument('--model', type=str, default="")
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
    num_test_runs = 15

    print('Initializing agent')
    try:
        tf.get_default_session().close()
    except AttributeError:
        pass
    sliding_target_updates = False
    agent = get_agent(args.agentname, env, log_dir, args.learning_rate, args.regularization_beta)
    stop_training = False
    if args.model is not "":
        agent.load_model(args.model)
        stop_training=True
    learning_rate = agent.model.learning_rate
    reg_beta = agent.model.beta
    if args.random_cartpole:
        args.envname = "CartPole-v1-random"
    print('Initialized')
    sarslist = []
    test_results = []
    global_steps = 0
    for i in range(args.num_rollouts):
        state = env.reset()
        done = False
        totalr = 0.
        steps = 0
        mean_cb_r = 0
        while not done:
            action = agent.get_action(state)
            log_action = action
            if args.random_cartpole and (state[0] > 0.2):
                action = env.action_space.sample()
            obs, r, done, _ = env.step(action)
            #if done and args.envname[:8] == "CartPole":
            #    r = -1
                # obs = np.zeros(env.observation_space.shape[0])

            sars = (state,
                    log_action,
                    obs,
                    r,
                    done)
            sarslist.append(sars)
            agent.replay_memory.append(sars)

            state = obs
            if len(agent.replay_memory) > agent.replay_memory_size:
                agent.replay_memory.pop(0)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
            count = 0
            if sliding_target_updates:
                current_weights = agent.model.get_weights()
                for w in agent.old_weights:
                    agent.old_weights[count] = np.add(np.multiply(agent.old_weights[count], 0.999),
                                                      np.multiply(current_weights[count], (1-0.999)))
                    if agent.old_weights[count].shape[0] == 1:
                        agent.old_weights[count] = agent.old_weights[count].reshape([-1])
                    count += 1
            elif global_steps % agent.target_update_freq == 0:
                current_weights = agent.model.get_weights()
                for w in agent.old_weights:
                    agent.old_weights[count] = np.array(current_weights[count])
                    if agent.old_weights[count].shape[0] == 1:
                        agent.old_weights[count] = agent.old_weights[count].reshape([-1])
                    count += 1
            if len(agent.replay_memory) > agent.minibatch_size and not stop_training:
                agent.train(args.no_tf_log)
                global_steps += 1
        returns.append(totalr)
        if i % (args.num_rollouts / 10) == 0:
            agent.save_model(log_dir, "{}_percent.ckpt".format(i / (args.num_rollouts / 100)))
        #if i % (args.num_rollouts / 100) == 0:
            # totalr = 0.
            # for j in range(num_test_runs):
            #     state = env.reset()
            #     action = env.action_space.sample()
            #     obs, r, done, _ = env.step(action)
            #     last_state = state
            #     state = obs
            #     while not done:
            #         action = agent.get_action(state, is_test=True)
            #         if args.random_cartpole and (state[0] > 0.2):
            #             action = env.action_space.sample()
            #         obs, r, done, _ = env.step(action)
            #         if done and args.envname[:8] == "CartPole":
            #             r = 0
            #         last_state = state
            #         state = obs
            #         totalr += r
            #         env.render()
            # test_results.append(totalr / num_test_runs)
            print("iter {0}, reward: {1:.2f}".format(i, totalr))
        #else:
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
    with open("{0}/trajectories.txt".format(log_dir), "w+") as f:
        f.write(str(sarslist))
