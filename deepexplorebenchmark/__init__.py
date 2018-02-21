from gym.envs.registration import register

register(
    id='DeepExploreBenchmark-v0',
    entry_point='deepexplorebenchmark.deepexplorebenchmark:DeepExploreBenchmark',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 19},
)

register(
    id='MountainCarStochasticArea-v0',
    entry_point='mountaincarstochasticarea.mountaincarstochasticarea:MountainCarEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 200},
)

register(
    id='CartPoleStochasticArea-v0',
    entry_point='cartpolestochasticarea.cartpolestochasticarea:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 200},
)
