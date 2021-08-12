from gym.envs.registration import register

register(
    id="Domus-v0",
    entry_point="domus_gym.envs:DomusEnv",
)

register(
    id="DomusCont-v0",
    entry_point="domus_gym.envs:DomusContEnv",
)
