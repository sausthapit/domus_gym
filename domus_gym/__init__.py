from gym.envs.registration import register

register(
    id="domus-v0",
    entry_point="domus_gym.envs:DomusEnv",
)
