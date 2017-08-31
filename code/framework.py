import pickle
import go_vncdriver
import tensorflow as tf
import numpy as np
import random
from basic_q_learning import Qlearner, Random_agent
import matplotlib.pyplot as plt

def get_agent(name, env):
    if name == "Qlearner":
        return Qlearner("Qlearner", env)
    elif name == "Random_agent":
        return Random_agent("Random_agent", env)
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
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    with tf.Session():

        import gym
        import universe
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        print('Initializing agent')
        agent = get_agent(args.agentname, env)
        print('Initialized')
        rewards = []
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            if len(returns) > 10:
                print("iter {0}, reward: {1:.2f}".format(i, returns[-1]))
            obs = env.reset()
            local_observations = []
            local_actions = []
            local_rewards = []
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = agent.get_action(obs[None, :])
                observations.append(obs)
                actions.append(action)
                local_observations.append(obs)
                local_actions.append(action)
                obs, r, done, _ = env.step(action)
                if done:
                    r = -100
                local_rewards.append(r)
                rewards.append(r)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps >= max_steps:
                    break
            if i > (args.num_rollouts / 10):
                agent.train(observations[-(args.num_rollouts/10):],
                            actions[-(args.num_rollouts/10):],
                            rewards[-(args.num_rollouts/10):])
            returns.append(totalr)
        plt.scatter(range(len(returns)), returns)
        plt.show()
