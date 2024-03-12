import gymnasium as gym

gym.register(
    id="Domus-v0",
    entry_point="domus_gym.envs:DomusEnv",
)

gym.register(
    id="DomusCont-v0",
    entry_point="domus_gym.envs:DomusContEnv",
)

gym.register(
    id="DomusFull-v0",
    entry_point="domus_gym.envs:DomusFullEnv",
)

gym.register(
    id="DomusFullAct-v0",
    entry_point="domus_gym.envs:DomusFullActEnv",
)

gym.register(
    id="DomusDv0Cont-v0",
    entry_point="domus_gym.envs:DomusDv0ContEnv",
)
