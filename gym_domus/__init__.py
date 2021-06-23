from gym.envs.registration import register

register(
    id='domus-v0',
    entry_point='gym_domus.envs:DomusEnv',
)
