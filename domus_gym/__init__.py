from gym.envs.registration import register

register(
    id="Domus-v0",
    entry_point="domus_gym.envs:DomusEnv",
)

register(
    id="DomusCont-v0",
    entry_point="domus_gym.envs:DomusContEnv",
)

register(
    id="DomusFull-v0",
    entry_point="domus_gym.envs:DomusFullEnv",
)

register(
    id="DomusFullAct-v0",
    entry_point="domus_gym.envs:DomusFullActEnv",
)
