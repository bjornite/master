from gym.envs.registration import register

register(
    id='DeepExploreBenchmark-v0',
    entry_point='deepexplorebenchmark.deepexplorebenchmark:DeepExploreBenchmark',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 19},
)
